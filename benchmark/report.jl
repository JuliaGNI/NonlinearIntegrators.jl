# Aggregate all per-problem benchmark CSVs into a single combined report.
#
#   julia --project=benchmark benchmark/report.jl
#
# Reads every results/*.csv (skipping already-combined outputs), writes a combined
# markdown report + plots covering all problems present. Decoupled from the sweep, so
# it can be re-run cheaply whenever new CSVs appear.

include(joinpath(@__DIR__, "gml_report.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "results")

function main()
    isdir(RESULTS_DIR) || error("no results directory at $(RESULTS_DIR); run a benchmark first")
    csvs = filter(f -> endswith(f, ".csv"), readdir(RESULTS_DIR; join = true))
    isempty(csvs) && error("no CSV files in $(RESULTS_DIR); run a benchmark first")
    println("Aggregating $(length(csvs)) CSV file(s):")
    foreach(c -> println("  ", basename(c)), csvs)

    rows = read_results(csvs)
    isempty(rows) && error("CSV files contained no data rows")

    md = write_report(rows;
        title = "One-layer GML benchmark — combined report",
        mode = "combined", outdir = RESULTS_DIR, prefix = "onelayer_gml_benchmark")
    println("Wrote combined report: ", md)
end

main()
