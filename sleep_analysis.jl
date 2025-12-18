begin
    using Pkg
    Pkg.activate(".")
end

packages = ["CSV", "DataFrames", "Dates", "PlotlyJS", "Statistics"];
importPackages = []
for p in packages
    p ∉ keys(Pkg.project().dependencies) && Pkg.add(p)
    eval(Meta.parse("using $p"))
end
for p in importPackages
    p ∉ keys(Pkg.project().dependencies) && Pkg.add(p)
    eval(Meta.parse("import $p"))
end

# Parse command line arguments
input_csv = "csv_files/Sleep Analysis.csv"
ref_bedtime_str = "23:00"

i = 1
while i <= length(ARGS)
    if ARGS[i] == "--bedtime" && i < length(ARGS)
        global ref_bedtime_str = ARGS[i+1]
        global i += 2
    elseif !startswith(ARGS[i], "-")
        global input_csv = ARGS[i]
        global i += 1
    else
        global i += 1
    end
end

const CSV_FILE = input_csv
# Parse reference bedtime to hour offset (e.g. 23:00 -> -1.0)
ref_hour, ref_min = parse.(Int, split(ref_bedtime_str, ":"))
ref_offset = ref_hour + ref_min / 60.0
if ref_offset > 12
    ref_offset -= 24.0
end

# 1. Load Data
if !isfile(CSV_FILE)
    println("Error: File $CSV_FILE not found.")
    exit(1)
end

println("Loading data from $CSV_FILE...")
println("Reference bedtime: $ref_bedtime_str (Offset: $ref_offset)")
df = CSV.read(CSV_FILE, DataFrame)

# 2. Preprocess Data
# Convert Start column to DateTime
# Format in CSV seems to be: "2025-12-09 22:50:35" (yyyy-mm-dd HH:MM:SS)
# CSV.read usually auto-detects, but let's ensure it.
if eltype(df.Start) <: AbstractString
    df.Start = DateTime.(String.(df.Start), "yyyy-mm-dd HH:MM:SS")
end

# Define 'Sleep Day' Logic:
# The day starts at 18:00 the previous day. 
# Meaning: 12/16 18:00 -> Belongs to 12/17.
# Logic: If we add 6 hours to the time:
#   12/16 18:00 + 6h = 12/17 00:00 -> Date is 12/17. Correct.
#   12/17 05:00 + 6h = 12/17 11:00 -> Date is 12/17. Correct.
#   12/17 17:59 + 6h = 12/17 23:59 -> Date is 12/17. Correct.
#   12/17 18:00 + 6h = 12/18 00:00 -> Date is 12/18. Correct.
df.SleepDay = Date.(df.Start .+ Hour(6))

# 3. Aggregate Data
# We want core, deep, rem, awake per day.
# Group by SleepDay and Value (Category)
gdf = groupby(df, [:SleepDay, :Value])
aggregated_df = combine(gdf, "Duration (hr)" => sum => :TotalDuration)

# Pivot (unstack) to get categories as columns
# Columns: SleepDay, Awake, Core, Deep, REM
final_df = unstack(aggregated_df, :SleepDay, :Value, :TotalDuration)

# Fill missing values with 0.0 (in case a category is missing for a day)
for col in names(final_df)
    if col != "SleepDay"
        final_df[!, col] = coalesce.(final_df[!, col], 0.0)
    end
end

# Calculate Total Sleep Time (Sum of all columns except SleepDay and Awake)
# We filter columns that are actually sleep stages (not metadata, not Awake)
# Note: Added potential other names just in case, but usually it's Core, Deep, REM. 
# Better approach: All columns excluding SleepDay and Awake.
cols_to_sum = filter(c -> c != "SleepDay" && c != "Awake", names(final_df))
final_df.TotalSleepTime = sum.(eachrow(final_df[:, cols_to_sum]))

# Sort by date just to be sure
sort!(final_df, :SleepDay)

# 3b. Calculate Bedtimes
# Group by SleepDay to find the earliest Start time (Bedtime)
gdf_day = groupby(df, :SleepDay)
bedtime_df = combine(gdf_day, :Start => minimum => :Bedtime)

# Calculate offset from midnight (in hours)
# If Bedtime is 23:00 previous day, SleepDay starts at 00:00. Difference is -1 hour.
bedtime_df.OffsetHours = [(Dates.value(b) - Dates.value(DateTime(d))) / 3_600_000 for (b, d) in zip(bedtime_df.Bedtime, bedtime_df.SleepDay)]

# calcuate my wake up time -- this is the last latest end date for each day, calculate it as hours past midnight
bedtime_df.WakeTime = [maximum(df.End[df.SleepDay.==d]) for d in bedtime_df.SleepDay]
# Ensure End column is DateTime for calculations
if eltype(df.End) <: AbstractString
    df.End = DateTime.(String.(df.End), "yyyy-mm-dd HH:MM:SS")
end

# Recalculate WakeTime to ensure they are DateTime objects
bedtime_df.WakeTime = [maximum(df.End[df.SleepDay.==d]) for d in bedtime_df.SleepDay]

# Calculate WakeTime offset from midnight (hours past midnight)
bedtime_df.WakeTimeOffsetHours = [(w - DateTime(d)) / Hour(1) for (w, d) in zip(bedtime_df.WakeTime, bedtime_df.SleepDay)]

# Sort by date
sort!(bedtime_df, :SleepDay)




# add bedtime to final_df
final_df.Bedtime = bedtime_df.OffsetHours
final_df.WakeTime = bedtime_df.WakeTimeOffsetHours

# remove any rows where bedtime is outside the reference window (+/- 2 hours)
# This assumes a 4-hour window around the expected bedtime.
final_df = final_df[final_df.Bedtime.>=(ref_offset-2.5), :]
final_df = final_df[final_df.Bedtime.<=(ref_offset+2.5), :]


#bedtime_sleep_correlation = cor(final_df.TotalSleepTime, final_df.Bedtime)
#println("Correlation between Bedtime (offset) and Total Sleep Time: ", round(bedtime_sleep_correlation, digits=4))

#rem_sleep_correlation = cor(final_df.REM, final_df.Bedtime)
#println("Correlation between REM and Bedtime: ", round(rem_sleep_correlation, digits=4))

#awake_sleep_correlation = cor(final_df.Awake, final_df.Bedtime)
#println("Correlation between Awake and Bedtime: ", round(awake_sleep_correlation, digits=4))

#deep_sleep_correlation = cor(final_df.Deep, final_df.Bedtime)
#println("Correlation between Deep and Bedtime: ", round(deep_sleep_correlation, digits=4))

#core_sleep_correlation = cor(final_df.Core, final_df.Bedtime)
#println("Correlation between Core and Total Sleep Time: ", round(core_sleep_correlation, digits=4))
# test the correlation of wake time with total sleep time
#wake_time_sleep_correlation = cor(final_df.TotalSleepTime, final_df.WakeTime)
#println("Correlation between Wake Time and Total Sleep Time: ", round(wake_time_sleep_correlation, digits=4))

# plot the distribution of bedtime and wake time
p_bedtime_dist = PlotlyJS.plot(
    PlotlyJS.histogram(x=final_df.Bedtime, nbinsx=20),
    PlotlyJS.Layout(title="Bedtime Distribution", xaxis_title="Hours from Midnight", yaxis_title="Count")
)
#display(p_bedtime_dist)

p_waketime_dist = PlotlyJS.plot(
    PlotlyJS.histogram(x=final_df.WakeTime, nbinsx=20),
    PlotlyJS.Layout(title="Wake Time Distribution", xaxis_title="Hours from Midnight", yaxis_title="Count")
)
#display(p_waketime_dist)

# calculate the standard deviation of sleep time and wake time
bedtime_std = std(final_df.Bedtime)
bedtime_mean = mean(final_df.Bedtime)
wake_time_std = std(final_df.WakeTime)
wake_time_mean = mean(final_df.WakeTime)
println("Standard Deviation of bedtime: ", round(bedtime_std, digits=4))
println("Standard Deviation of Wake Time: ", round(wake_time_std, digits=4))

# Using statsplots make a full correlation matrix of everything but Sleep Day in final_df

# Using PlotlyJS to make a full correlation matrix of everything but Sleep Day in final_df

corr_cols = filter(c -> c ∉ ["SleepDay", "Asleep"], names(final_df))
# Create a sub-dataframe with only the numeric columns needed for correlation
sub_df = final_df[:, corr_cols]
# Compute correlation matrix
cor_matrix = cor(Matrix(sub_df))

# Create Annotations for Heatmap
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

# Create Heatmap
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


PlotlyJS.savefig(p_corr, "correlation_matrix.html")
println("Correlation matrix saved to correlation_matrix.html")
# 4. Generate Stacked Bar Chart & Bedtime Plot

# Top Chart: Sleep Composition (Stacked Bar)
categories = ["Deep", "Core", "REM", "Awake"]
bar_traces = PlotlyJS.GenericTrace[]

existing_categories = filter(c -> c in names(final_df), categories)

for cat in existing_categories
    push!(bar_traces, PlotlyJS.bar(
        x=final_df.SleepDay,
        y=final_df[!, cat],
        name=cat,
        xaxis="x", # Shared axis
        yaxis="y"  # Top plot y-axis
    ))
end

# Bottom Chart: Bedtime (Scatter)
bedtime_trace = PlotlyJS.scatter(
    x=bedtime_df.SleepDay,
    y=bedtime_df.OffsetHours,
    mode="lines+markers",
    name="Bedtime",
    line=PlotlyJS.attr(color="navy", width=2),
    marker=PlotlyJS.attr(size=6),
    xaxis="x", # Shared axis
    yaxis="y2" # Bottom plot y-axis
)

# Combine Traces
all_traces = [bar_traces; bedtime_trace]

# Define Layout with Subplots
layout = PlotlyJS.Layout(
    title="Daily Sleep Analysis",
    barmode="stack",
    grid=PlotlyJS.attr(
        rows=2,
        columns=1,
        pattern="independent",
        roworder="top to bottom"
    ),
    # X-axis configuration (Shared)
    xaxis=PlotlyJS.attr(
        title="Date",
        anchor="y2" # Anchored to bottom plot
    ),
    # Y-axis 1 (Top: Duration)
    yaxis=PlotlyJS.attr(
        title="Duration (Hours)",
        domain=[0.35, 1.0] # Top 65% of area
    ),
    # Y-axis 2 (Bottom: Bedtime)
    yaxis2=PlotlyJS.attr(
        title="Bedtime (Hrs vs Midnight)",
        domain=[0.0, 0.25] # Bottom 25% of area
    ),
    height=800,
    showlegend=true
)

p = PlotlyJS.plot(all_traces, layout)



# Save to html for reference
safesave_path = "sleep_analysis_plot.html"
PlotlyJS.savefig(p, safesave_path)
println("Plot saved to $safesave_path")
