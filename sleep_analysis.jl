begin
    using Pkg
    Pkg.activate(".")
end

packages = ["CSV", "DataFrames", "Dates", "PlotlyJS", "Statistics", "GLM", "GoogleSheets"];
importPackages = []
for p in packages
    p ∉ keys(Pkg.project().dependencies) && Pkg.add(p)
    eval(Meta.parse("using $p"))
end
for p in importPackages
    p ∉ keys(Pkg.project().dependencies) && Pkg.add(p)
    eval(Meta.parse("import $p"))
end

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
    format_hour_offset(offset::Real) -> String

Convert hour offset from midnight to HH:MM format.
Negative offsets wrap around (e.g., -1.0 -> "23:00").
"""
function format_hour_offset(offset::Real)
    h = offset
    while h < 0
        h += 24
    end
    while h >= 24
        h -= 24
    end

    hours = floor(Int, h)
    minutes = round(Int, (h - hours) * 60)

    if minutes == 60
        hours += 1
        minutes = 0
    end

    hours = hours % 24

    return lpad(hours, 2, '0') * ":" * lpad(minutes, 2, '0')
end

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

"""
    parse_arguments() -> NamedTuple

Parse command-line arguments for input CSV file, reference bedtime, and Google Sheets options.
Returns: (csv_file, ref_bedtime_str, ref_offset, sheet_id, sheet_name)
"""
function parse_arguments()
    input_csv = nothing  # Will be determined automatically if not specified
    ref_bedtime_str = "23:00"
    sheet_id = nothing  # Optional Google Sheets spreadsheet ID
    sheet_name = "Sheet1"  # Default worksheet name

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--bedtime" && i < length(ARGS)
            ref_bedtime_str = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--sheet-id" && i < length(ARGS)
            sheet_id = ARGS[i+1]
            i += 2
        elseif ARGS[i] == "--sheet-name" && i < length(ARGS)
            sheet_name = ARGS[i+1]
            i += 2
        elseif !startswith(ARGS[i], "-")
            input_csv = ARGS[i]
            i += 1
        else
            i += 1
        end
    end

    # If no CSV specified, find the most recently modified CSV in csv_files/
    if isnothing(input_csv)
        csv_dir = "csv_files"
        if isdir(csv_dir)
            csv_files = filter(f -> endswith(lowercase(f), ".csv"), readdir(csv_dir))
            if !isempty(csv_files)
                # Get modification times and find most recent
                csv_with_times = [(f, mtime(joinpath(csv_dir, f))) for f in csv_files]
                sort!(csv_with_times, by=x -> x[2], rev=true)
                input_csv = joinpath(csv_dir, csv_with_times[1][1])
                println("Auto-selected most recent CSV: $input_csv")
            else
                println("No CSV files found in $csv_dir")
                input_csv = "csv_files/Sleep Analysis.csv"  # fallback
            end
        else
            input_csv = "csv_files/Sleep Analysis.csv"  # fallback
        end
    end

    # Parse reference bedtime to hour offset (e.g. 23:00 -> -1.0)
    ref_hour, ref_min = parse.(Int, split(ref_bedtime_str, ":"))
    ref_offset = ref_hour + ref_min / 60.0
    if ref_offset > 12
        ref_offset -= 24.0
    end

    return (input_csv, ref_bedtime_str, ref_offset, sheet_id, sheet_name)
end

"""
    load_and_preprocess_data(csv_file::String) -> DataFrame

Load CSV file and convert Start and End columns to DateTime.
"""
function load_and_preprocess_data(csv_file::String)
    if !isfile(csv_file)
        println("Error: File $csv_file not found.")
        exit(1)
    end

    println("Loading data from $csv_file...")
    df = CSV.read(csv_file, DataFrame)

    # Convert Start column to DateTime if needed
    if eltype(df.Start) <: AbstractString
        df.Start = DateTime.(String.(df.Start), "yyyy-mm-dd HH:MM:SS")
    end

    # Convert End column to DateTime if needed
    if eltype(df.End) <: AbstractString
        df.End = DateTime.(String.(df.End), "yyyy-mm-dd HH:MM:SS")
    end

    # every time you see 'Awake' duplicate that row but change the value to AwakeCount and the "Duration (hr)" to 1
    dfAdd = DataFrame()
    for i in 1:nrow(df)
        # print out the row 
        #println(df[i, :])
        if df.Value[i] == "Awake"
            push!(dfAdd, df[i, :], promote=true)
        end
    end
    dfAdd.Value .= "AwakeCount"
    dfAdd[!, "Duration (hr)"] .= 1.0
    append!(df, dfAdd, promote=true)

    return df
end

"""
    calculate_sleep_days!(df::DataFrame)

Add SleepDay column based on 6-hour offset logic.
Sleep day starts at 18:00 the previous calendar day.
Modifies DataFrame in-place.
"""
function calculate_sleep_days!(df::DataFrame)
    # Define 'Sleep Day' Logic:
    # The day starts at 18:00 the previous day. 
    # Meaning: 12/16 18:00 -> Belongs to 12/17.
    # Logic: If we add 6 hours to the time:
    #   12/16 18:00 + 6h = 12/17 00:00 -> Date is 12/17. Correct.
    #   12/17 05:00 + 6h = 12/17 11:00 -> Date is 12/17. Correct.
    #   12/17 17:59 + 6h = 12/17 23:59 -> Date is 12/17. Correct.
    #   12/17 18:00 + 6h = 12/18 00:00 -> Date is 12/18. Correct.
    df.SleepDay = Date.(df.Start .+ Hour(6))
end

"""
    aggregate_sleep_data(df::DataFrame) -> DataFrame

Aggregate sleep data by SleepDay and sleep stage (Value).
Returns unstacked DataFrame with columns: SleepDay, Core, Deep, REM, Awake, TotalSleepTime
"""
function aggregate_sleep_data(df::DataFrame)
    # Group by SleepDay and Value (Category)
    gdf = groupby(df, [:SleepDay, :Value])
    aggregated_df = combine(gdf, "Duration (hr)" => sum => :TotalDuration)
    # after each entry for 'awake' add a new row with teh same SleepDay but named AwakeCount which is the sum of AwakeCount in df on that sleep day


    # Pivot (unstack) to get categories as columns
    final_df = unstack(aggregated_df, :SleepDay, :Value, :TotalDuration)

    # Drop the Asleep column if it exists
    select!(final_df, setdiff(names(final_df), ["Asleep"]))

    # Fill missing values with 0.0 (in case a category is missing for a day)
    for col in names(final_df)
        if col != "SleepDay"
            final_df[!, col] = coalesce.(final_df[!, col], 0.0)
        end
    end

    # Calculate Total Sleep Time (Sum of all columns except SleepDay and Awake)
    cols_to_sum = filter(c -> c != "SleepDay" && c != "Awake" && c != "AwakeCount", names(final_df))
    final_df.TotalSleepTime = sum.(eachrow(final_df[:, cols_to_sum]))

    # Sort by date
    sort!(final_df, :SleepDay)

    return final_df
end

"""
    calculate_bedtime_waketime(df::DataFrame) -> DataFrame

Calculate bedtime (earliest Start) and waketime (latest End) for each SleepDay.
Returns DataFrame with columns: SleepDay, BedTime, OffsetHours, WakeTime, WakeTimeOffsetHours
"""
function calculate_bedtime_waketime(df::DataFrame)
    # Group by SleepDay to find the earliest Start time (Bedtime)
    gdf_day = groupby(df, :SleepDay)
    bedtime_df = combine(gdf_day, :Start => minimum => :BedTime)

    # Calculate offset from midnight (in hours)
    # If Bedtime is 23:00 previous day, SleepDay starts at 00:00. Difference is -1 hour.
    bedtime_df.OffsetHours = [(Dates.value(b) - Dates.value(DateTime(d))) / 3_600_000
                              for (b, d) in zip(bedtime_df.BedTime, bedtime_df.SleepDay)]

    # Calculate wake up time (latest End time for each day)
    bedtime_df.WakeTime = [maximum(df.End[df.SleepDay.==d]) for d in bedtime_df.SleepDay]

    # Calculate WakeTime offset from midnight (hours past midnight)
    bedtime_df.WakeTimeOffsetHours = [(w - DateTime(d)) / Hour(1)
                                      for (w, d) in zip(bedtime_df.WakeTime, bedtime_df.SleepDay)]

    # calculate the number of times I was Awake

    # Sort by date
    sort!(bedtime_df, :SleepDay)

    return bedtime_df
end

"""
    merge_and_filter_data(final_df::DataFrame, bedtime_df::DataFrame, ref_offset::Float64) -> DataFrame

Merge bedtime/waketime data into final_df and filter outliers.
Filters out rows with missing Core/Deep/REM and bedtimes outside ref_offset ± 2.5 hours.
"""
function merge_and_filter_data(final_df::DataFrame, bedtime_df::DataFrame, ref_offset::Float64)
    # Add bedtime and waketime to final_df
    final_df.BedTime = bedtime_df.OffsetHours
    final_df.WakeTime = bedtime_df.WakeTimeOffsetHours
    #    bedtime_df.AwakeCount = [sum(df.Awake[df.SleepDay.==d]) for d in bedtime_df.SleepDay]
    #final_df.AwakeCount = bedtime_df.AwakeCount

    # Drop any rows that have missing data in :Core, :Deep, :REM columns
    final_df = dropmissing(final_df, [:Core, :Deep, :REM])

    # Remove any rows where bedtime is outside the reference window (± 2.5 hours)
    final_df = final_df[final_df.BedTime.>=(ref_offset-2.5), :]
    final_df = final_df[final_df.BedTime.<=(ref_offset+2.5), :]

    return final_df
end

"""
    create_export_dataframe(final_df::DataFrame) -> DataFrame

Create formatted DataFrame for CSV export.
Converts time offsets to HH:MM format, rounds numeric columns, and reorders columns.
"""
function create_export_dataframe(final_df::DataFrame)
    # Create a copy for export
    export_df = copy(final_df)

    # Format bedtime and waketime as HH:MM strings
    export_df.BedTime = format_hour_offset.(final_df.BedTime)
    export_df.WakeTime = format_hour_offset.(final_df.WakeTime)

    # Round all numeric columns to 2 decimals
    for col in names(export_df)
        if col != "SleepDay" && col != "BedTime" && col != "WakeTime"
            export_df[!, col] = round.(export_df[!, col], digits=2)
        end
    end

    # Reorder columns for export
    export_df = export_df[:, [:SleepDay, :Deep, :Core, :REM, :Awake, :AwakeCount, :TotalSleepTime, :BedTime, :WakeTime]]

    return export_df
end

"""
    generate_visualizations(final_df::DataFrame, bedtime_df::DataFrame)

Generate and save all visualizations:
- Bedtime and waketime distribution histograms
- Correlation matrix heatmap
- Stacked bar chart with bedtime overlay
"""
function generate_visualizations(final_df::DataFrame, bedtime_df::DataFrame)
    # Create results directory if it doesn't exist
    results_dir = "results"
    if !isdir(results_dir)
        mkdir(results_dir)
        println("Created results directory: $results_dir")
    end

    # Calculate and print statistics
    bedtime_std = std(final_df.BedTime)
    bedtime_mean = mean(final_df.BedTime)
    wake_time_std = std(final_df.WakeTime)
    wake_time_mean = mean(final_df.WakeTime)
    println("Standard Deviation of bedtime: ", round(bedtime_std, digits=4))
    println("Standard Deviation of Wake Time: ", round(wake_time_std, digits=4))

    # Bedtime distribution histogram
    p_bedtime_dist = PlotlyJS.plot(
        PlotlyJS.histogram(x=final_df.BedTime, nbinsx=20),
        PlotlyJS.Layout(title="Bedtime Distribution", xaxis_title="Hours from Midnight", yaxis_title="Count")
    )
    PlotlyJS.savefig(p_bedtime_dist, joinpath(results_dir, "bedtime_distribution.html"))
    println("Bedtime distribution saved to $(joinpath(results_dir, "bedtime_distribution.html"))")

    # Waketime distribution histogram
    p_waketime_dist = PlotlyJS.plot(
        PlotlyJS.histogram(x=final_df.WakeTime, nbinsx=20),
        PlotlyJS.Layout(title="Wake Time Distribution", xaxis_title="Hours from Midnight", yaxis_title="Count")
    )
    PlotlyJS.savefig(p_waketime_dist, joinpath(results_dir, "waketime_distribution.html"))
    println("Waketime distribution saved to $(joinpath(results_dir, "waketime_distribution.html"))")

    # Correlation matrix
    corr_cols = filter(c -> c ∉ ["SleepDay", "Asleep"], names(final_df))
    sub_df = final_df[:, corr_cols]
    cor_matrix = cor(Matrix(sub_df))

    # Create annotations for heatmap
    annotations = [
        PlotlyJS.attr(
            x=corr_cols[j],
            y=corr_cols[i],
            text=string(round(cor_matrix[i, j], digits=2)),
            showarrow=false,
            font=PlotlyJS.attr(color="black"),
            bgcolor="rgba(240, 240, 240, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=3
        )
        for i in 1:size(cor_matrix, 1) for j in 1:size(cor_matrix, 2)
    ]

    p_corr = PlotlyJS.plot(
        PlotlyJS.heatmap(
            z=cor_matrix,
            x=corr_cols,
            y=corr_cols,
            colorscale=:diverging_bwr_20_95_c54_n256,
            reversescale=false
        ),
        PlotlyJS.Layout(
            title="Sleep Metrics Correlation Matrix",
            width=800,
            height=800,
            xaxis_side="bottom",
            annotations=annotations
        )
    )

    PlotlyJS.savefig(p_corr, joinpath(results_dir, "correlation_matrix.html"))
    println("Correlation matrix saved to $(joinpath(results_dir, "correlation_matrix.html"))")

    # Stacked bar chart with bedtime overlay
    categories = ["Deep", "Core", "REM", "Awake", "AwakeCount"]
    bar_traces = PlotlyJS.GenericTrace[]

    existing_categories = filter(c -> c in names(final_df), categories)

    for cat in existing_categories
        push!(bar_traces, PlotlyJS.bar(
            x=final_df.SleepDay,
            y=final_df[!, cat],
            name=cat,
            xaxis="x",
            yaxis="y"
        ))
    end

    # Bottom chart: Bedtime scatter
    bedtime_trace = PlotlyJS.scatter(
        x=bedtime_df.SleepDay,
        y=bedtime_df.OffsetHours,
        mode="lines+markers",
        name="Bedtime",
        line=PlotlyJS.attr(color="navy", width=2),
        marker=PlotlyJS.attr(size=6),
        xaxis="x",
        yaxis="y2"
    )

    all_traces = [bar_traces; bedtime_trace]

    layout = PlotlyJS.Layout(
        title="Daily Sleep Analysis",
        barmode="stack",
        grid=PlotlyJS.attr(
            rows=2,
            columns=1,
            pattern="independent",
            roworder="top to bottom"
        ),
        xaxis=PlotlyJS.attr(
            title="Date",
            anchor="y2"
        ),
        yaxis=PlotlyJS.attr(
            title="Duration (Hours)",
            domain=[0.35, 1.0]
        ),
        yaxis2=PlotlyJS.attr(
            title="Bedtime (Hrs vs Midnight)",
            domain=[0.0, 0.25]
        ),
        height=800,
        showlegend=true
    )

    p = PlotlyJS.plot(all_traces, layout)
    PlotlyJS.savefig(p, joinpath(results_dir, "sleep_analysis_plot.html"))
    println("Plot saved to $(joinpath(results_dir, "sleep_analysis_plot.html"))")
end

# ==========================================
# 2. Analysis & Insight Generation
# ==========================================

function parse_bedtime(val)
    try
        h, m = parse.(Int, split(t_str, ":"))
        val = h + m / 60.0
        # If time is after 6 PM (18:00), treat it as previous day (negative relative to midnight)
        if val >= 18.0
            return val - 24.0
        end
        return val
    catch
        return missing
    end
end


function analyze_sleep_data(df::DataFrame)

    # Create results directory if it doesn't exist
    results_dir = "results"
    if !isdir(results_dir)
        mkdir(results_dir)
        println("Created results directory: $results_dir")
    end

    # Create a clean numeric column for Bedtime 
    # make a copy of df

    df.BedTime_Float = df.BedTime
    dfAnalysis = copy(df)
    dfAnalysis.DeepCore = dfAnalysis.Core .+ dfAnalysis.Deep

    targets = ["TotalSleepTime", "Core", "REM", "Deep", "Awake", "DeepCore", "AwakeCount"]
    plots = GenericTrace[]

    println("--------------- HUMAN READABLE INSIGHTS ---------------")
    println("Baseline: Analyzing how 1 hour of LATER bedtime affects sleep metrics.\n")

    for target in targets
        # Prepare data for this specific target
        # We use @formula macro: Target ~ BedTime
        # The term 'BedTime_Float' is our X, 'target' is our Y

        # Dynamic column access requires symbols
        col_sym = Symbol(target)

        # Fit the linear model
        # Note: We create a temporary view to handle missing values in specific columns if any
        data_clean = dropmissing(dfAnalysis[:, [:SleepDay, :BedTime_Float, col_sym]])
        ols = lm(term(col_sym) ~ term(:BedTime_Float), data_clean)

        # Extract coefficients
        intercept = coef(ols)[1]
        slope = coef(ols)[2]
        r2_val = r2(ols)

        # --- Generate Insights ---
        # Convert hours to minutes for readability
        minutes_change = round(slope * 60, digits=1)
        direction = slope < 0 ? "LOSS" : "GAIN"

        println("METRIC: $(uppercase(target))")
        println("  • Trend: $direction of $(abs(minutes_change)) minutes for every hour you delay bedtime.")
        println("  • Statistical Fit (R²): $(round(r2_val, digits=3))")

        if abs(r2_val) < 0.05
            println("  • INTERPRETATION: No significant link. Bedtime does not affect this metric.")
        elseif target == "TotalSleepTime"
            println("  • INTERPRETATION: Strong validation. Earlier bedtimes significantly increase total rest.")
        elseif target == "Core"
            println("  • INTERPRETATION: This is the main driver. Most gained sleep comes from this stage.")
        end
        println("-------------------------------------------------------")

        # --- Generate Plot ---
        # Define date boundaries for categories
        ambien_end = Date(2025, 8, 1)
        trazadone_end = Date(2025, 12, 13)

        # Split data into three categories
        data_ambien = data_clean[data_clean.SleepDay.<ambien_end, :]
        data_trazadone = data_clean[(data_clean.SleepDay.>=ambien_end).&(data_clean.SleepDay.<=trazadone_end), :]
        data_sleepstack = data_clean[data_clean.SleepDay.>trazadone_end, :]

        # Create scatter traces for each category
        scatter_ambien = scatter(
            x=data_ambien.BedTime_Float,
            y=data_ambien[!, col_sym],
            mode="markers",
            name="Ambien (before Aug 1)",
            marker=attr(opacity=0.6, size=8, color="blue")
        )

        scatter_trazadone = scatter(
            x=data_trazadone.BedTime_Float,
            y=data_trazadone[!, col_sym],
            mode="markers",
            name="Trazadone (Aug 1 - Dec 13)",
            marker=attr(opacity=0.6, size=8, color="green")
        )

        scatter_sleepstack = scatter(
            x=data_sleepstack.BedTime_Float,
            y=data_sleepstack[!, col_sym],
            mode="markers",
            name="Sleep Stack (after Dec 13)",
            marker=attr(opacity=0.6, size=8, color="purple")
        )

        # Create the regression line
        # We predict Y values using the model for the range of X
        x_range = range(minimum(data_clean.BedTime_Float), maximum(data_clean.BedTime_Float), length=100)
        y_pred = intercept .+ slope .* x_range

        line_trace = scatter(
            x=x_range,
            y=y_pred,
            mode="lines",
            name="Trend Line",
            line=attr(color="red", width=3)
        )

        # Create annotation text with R² and slope interpretation
        # Since earlier bedtime = more negative offset, and negative slope means more sleep with negative offset,
        # we need to interpret: for each hour EARLIER (i.e., more negative), we get (-slope * 60) more minutes
        r2_pct = round(r2_val * 100, digits=1)
        minutes_per_hour_earlier = round(-slope * 60, digits=1)  # negative slope means positive gain for earlier bedtime
        change_word = minutes_per_hour_earlier >= 0 ? "more" : "less"

        annotation_text = "$(r2_pct)% of the variation in $target is explained by bedtime.<br>" *
                          "For every hour earlier you go to bed,<br>" *
                          "you get $(abs(minutes_per_hour_earlier)) minutes $change_word of $target."

        # Combine into a plot object
        p = plot([scatter_ambien, scatter_trazadone, scatter_sleepstack, line_trace], Layout(
            title="Bedtime vs $target",
            xaxis_title="Bedtime (Hours from Midnight)",
            yaxis_title="$target (Hours)",
            hovermode="closest",
            template="plotly_white",
            annotations=[
                attr(
                    x=0.02,
                    y=0.02,
                    xref="paper",
                    yref="paper",
                    text=annotation_text,
                    showarrow=false,
                    font=attr(size=11, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=6,
                    align="left",
                    xanchor="left",
                    yanchor="bottom"
                )
            ]
        ))

        # Save the plot to results directory
        plot_filename = joinpath(results_dir, "bedtime_vs_$(lowercase(target)).html")
        PlotlyJS.savefig(p, plot_filename)
        println("Saved plot: $plot_filename")
    end
end

# ============================================================================
# GOOGLE SHEETS INTEGRATION
# ============================================================================

"""
    upload_to_google_sheets(export_df::DataFrame, spreadsheet_id::String, sheet_name::String="Sheet1")

Upload sleep data to an existing Google Sheet, matching dates in column A.
Requires credentials.json in ~/.julia/google_sheets/
"""
function upload_to_google_sheets(export_df::DataFrame, spreadsheet_id::String, sheet_name::String="Sheet1")
    println("Connecting to Google Sheets...")

    # Connect with read/write permissions
    client = GoogleSheets.sheets_client(GoogleSheets.AUTH_SCOPE_READWRITE)
    sheet = GoogleSheets.Spreadsheet(spreadsheet_id)

    # Read column A to get existing dates and find last row
    date_range = GoogleSheets.CellRange(sheet, "$(sheet_name)!A:A")
    date_result = GoogleSheets.get(client, date_range)

    # Find the last row with data and the last date
    last_row = 0
    last_date = nothing

    if !isnothing(date_result.values)
        for (idx, row) in enumerate(eachrow(date_result.values))
            if length(row) > 0 && !isempty(row[1])
                date_str = string(row[1])
                last_row = idx

                # Try to parse the date
                for fmt in ["yyyy-mm-dd", "m/d/yyyy", "mm/dd/yyyy"]
                    try
                        parsed_date = Dates.Date(date_str, fmt)
                        if isnothing(last_date) || parsed_date > last_date
                            last_date = parsed_date
                        end
                        break
                    catch
                        continue
                    end
                end
            end
        end
    end

    println("Last row with data: $last_row")
    if !isnothing(last_date)
        println("Last date in sheet: $last_date")
    end

    # Filter export_df to only include rows AFTER the last date
    if !isnothing(last_date)
        export_df_filtered = filter(row -> row.SleepDay > last_date, export_df)
        println("Rows to upload (after $last_date): $(nrow(export_df_filtered))")
    else
        export_df_filtered = export_df
        println("No valid dates found in sheet, uploading all rows.")
    end

    if nrow(export_df_filtered) == 0
        println("No new data to upload.")
        return
    end

    # Sort by date to ensure chronological order
    export_df_filtered = sort(export_df_filtered, :SleepDay)

    # Append each new row after the last row
    rows_added = 0
    current_row = last_row + 1

    for row in eachrow(export_df_filtered)
        println("Adding row $current_row: $(row.SleepDay)")

        # Prepare data for columns A through I
        # Order: SleepDay, Deep, Core, REM, Awake, AwakeCount, TotalSleepTime, BedTime, WakeTime
        values = [
            string(row.SleepDay),
            row.Deep,
            row.Core,
            row.REM,
            row.Awake,
            row.AwakeCount,
            row.TotalSleepTime,
            row.BedTime,
            row.WakeTime
        ]

        println("Values to upload: $values")
        # Write to row (columns A:I)
        cell_range = GoogleSheets.CellRange(sheet, "$(sheet_name)!A$current_row:I$current_row")
        GoogleSheets.update!(client, cell_range, reshape(values, 1, :))
        rows_added += 1
        current_row += 1
    end

    println("Google Sheets update complete: $rows_added rows added.")
end

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

"""
    main()

Main workflow orchestration function.
Executes the complete sleep analysis pipeline.
"""
#function main()
# 1. Parse command-line arguments
csv_file, ref_bedtime_str, ref_offset, sheet_id, sheet_name = parse_arguments()
println("Reference bedtime: $ref_bedtime_str (Offset: $ref_offset)")
if !isnothing(sheet_id)
    println("Google Sheets upload enabled: $sheet_id (Sheet: $sheet_name)")
end

# 2. Load and preprocess data
df = load_and_preprocess_data(csv_file)



# 3. Calculate sleep days
calculate_sleep_days!(df)

# 4. Aggregate sleep data
final_df = aggregate_sleep_data(df)

# 5. Calculate bedtime and waketime
bedtime_df = calculate_bedtime_waketime(df)

# 6. Merge and filter data
final_df = merge_and_filter_data(final_df, bedtime_df, ref_offset)

# 7. Create export dataframe and save to CSV
export_df = create_export_dataframe(final_df)
CSV.write("sleep_data.csv", export_df)
println("Sleep data saved to sleep_data.csv")

# 8. Upload to Google Sheets (if sheet_id provided)
if !isnothing(sheet_id)
    upload_to_google_sheets(export_df, sheet_id, sheet_name)
end

# 9. Generate visualizations
generate_visualizations(final_df, bedtime_df)
#end

analyze_sleep_data(final_df)

# Execute main workflow
#main()



