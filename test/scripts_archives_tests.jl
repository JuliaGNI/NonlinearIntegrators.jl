# The shared layer under every driver in `scripts/`: the option parser, the archive schema helpers
# and the failure reporter.
#
# Tested here rather than by hand-running a driver because the parser is the one gate between a
# mistyped flag and a twenty-minute sweep that runs the wrong thing, and because the schema helpers
# are what let a figure be redrawn from an archive written by an older revision — both are
# properties a driver only exercises after the solves have already been paid for.
#
# Loaded into a module of its own so that the script's globals — `RUNS_DIR`, `RESULTS_DIR`,
# `report`, `banner` — do not land in `Main` beside the rest of the suite.
module ScriptsArchivesTests

using Test

include(joinpath(@__DIR__, "..", "scripts", "archives.jl"))

@testset "scripts/archives.jl" begin
    @testset "parse_arguments" begin
        # The parser writes the output directories as a side effect, so the suite puts them back.
        runs_before, results_before = RUNS_DIR[], RESULTS_DIR[]
        try
            @testset "splits positionals from options" begin
                names, options = parse_arguments(
                    ["ho", "pendulum", "--steps", "1,2,5"], ("--steps",))
                @test names == ["ho", "pendulum"]
                @test options == Dict("--steps" => "1,2,5")
            end

            @testset "positionals may follow an option" begin
                names, _ = parse_arguments(["--steps", "1,2", "ho"], ("--steps",))
                @test names == ["ho"]
            end

            @testset "the common options are accepted everywhere and applied" begin
                names, _ = parse_arguments([
                    "--runs-dir", "/x/runs", "--results-dir", "/x/res"])
                @test isempty(names)
                @test RUNS_DIR[] == "/x/runs"
                @test RESULTS_DIR[] == "/x/res"
            end

            @testset "an unknown option is rejected, not absorbed" begin
                # The failure this replaces: a driver that pushed the flag onto its list of problem
                # names, matched nothing, and then ran the entire sweep.
                @test_throws ArgumentError parse_arguments(["--nonsense", "1"], ("--steps",))
                # A flag another driver knows is still unknown to one that does not declare it.
                @test_throws ArgumentError parse_arguments(["--final-time", "10"])
                @test_throws "--nonsense" parse_arguments(["--nonsense", "1"])
            end

            @testset "an option with no value is rejected" begin
                @test_throws ArgumentError parse_arguments(["--steps"], ("--steps",))
                @test_throws "--steps" parse_arguments(["ho", "--steps"], ("--steps",))
            end
        finally
            RUNS_DIR[], RESULTS_DIR[] = runs_before, results_before
        end
    end

    @testset "option_steps and option_final_time" begin
        @test option_steps(Dict("--steps" => "1,2,5"), (9.0,)) == (1.0, 2.0, 5.0)
        @test option_steps(Dict{String, String}(), (0.5, 0.25)) == (0.5, 0.25)
        @test option_final_time(Dict("--final-time" => "12.5"), 1.0) == 12.5
        @test option_final_time(Dict{String, String}(), 1000.0) == 1000.0
    end

    @testset "archive_kind" begin
        @test archive_kind(Dict{String, Any}("kind" => "solution")) == "solution"
        @test archive_kind(Dict{String, Any}("kind" => "convergence")) == "convergence"

        # An explicit `"kind"` is authoritative: it is read before the series are looked at.
        @test archive_kind(Dict{String, Any}("kind" => "solution",
            "timesteps" => [1.0], "errors" => [[1.0]])) == "solution"

        # Inference, which is what keeps archives written before `"kind"` existed drawable.
        @test archive_kind(Dict{String, Any}("timesteps" => [1.0],
            "errors" => [[1.0]])) == "convergence"
        @test archive_kind(Dict{String, Any}("t" => [0.0], "q" => [0.0],
            "p" => [0.0])) == "solution"

        # Neither shape, and in particular not a partial one.
        @test archive_kind(Dict{String, Any}("label" => "x")) === nothing
        @test archive_kind(Dict{String, Any}("t" => [0.0], "q" => [0.0])) === nothing
        @test archive_kind(Dict{String, Any}("timesteps" => [1.0])) === nothing
    end

    @testset "normalise_schema!" begin
        @testset "the old scalar spelling becomes the current vector one" begin
            data = Dict{String, Any}("figure_window" => 5.0)
            @test normalise_schema!(data) === data
            @test data["windows"] == [5.0]
        end

        @testset "an archive already carrying `windows` is left alone" begin
            data = Dict{String, Any}("figure_window" => 5.0, "windows" => [1.0, 2.0])
            normalise_schema!(data)
            @test data["windows"] == [1.0, 2.0]
        end

        @testset "an archive with neither key gains neither" begin
            data = Dict{String, Any}("stem" => "ho-nvi-h1.0")
            normalise_schema!(data)
            @test !haskey(data, "windows")
        end
    end

    @testset "failure_message" begin
        # The type alone is not enough: these three carry the key, the argument and the value that
        # say which archive failed and why.
        @test occursin("missing", failure_message(KeyError("missing")))
        @test occursin("bad step", failure_message(ArgumentError("bad step")))

        # One line, capped — a `MethodError` prints its whole candidate list, which does not belong
        # in a skip report.
        message = failure_message(MethodError(sin, (nothing,)))
        @test !occursin('\n', message)
        @test length(message) ≤ 161
    end
end

end # module
