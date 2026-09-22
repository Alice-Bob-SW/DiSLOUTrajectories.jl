using Test
using TestReports
using TestReports.EzXML: root, findall

function write_junit(document, output)
    # TestReports 1.4 emits <skip>; JUnit consumers expect <skipped>.
    for skipped in findall("//testcase/skip", root(document))
        skipped.name = "skipped"
    end
    open(output, "w") do io
        print(io, document)
    end
    return nothing
end

isempty(ARGS) && error("Usage: julia test/junit.jl --package | TEST_FILE [TEST_FILE ...]")
output = abspath(get(ENV, "JUNIT_OUTPUT", "junit.xml"))
if ARGS == ["--package"]
    testfile = joinpath(dirname(Base.active_project()), "test", "runtests.jl")
    coverage = Base.JLOptions().code_coverage != 0 ? "user" : "none"
    # Use the same nested reporting runner inside the isolated package test environment.
    TestReports.TestEnv.activate() do
        command = `$(Base.julia_cmd()) --project=$(Base.active_project())
                   --startup-file=no --check-bounds=yes --threads=$(Threads.nthreads())
                   --code-coverage=$coverage $(@__FILE__) $testfile`
        run(addenv(command, "JUNIT_OUTPUT" => output))
    end
else
    files = abspath.(ARGS)
    run_tests() = @testset ReportingTestSet "DiSLOUTrajectories" begin
        for file in files
            include(file)
        end
    end
    # Keep the testset nested: TestReports 1.4's top-level display mutates
    # DefaultTestSet fields that are const on Julia 1.13.
    results = @static if VERSION >= v"1.13"
        Test.@with_testset ReportingTestSet("JUnit") run_tests()
    else
        Test.push_testset(ReportingTestSet("JUnit"))
        try
            run_tests()
        finally
            Test.pop_testset()
        end
    end
    # JET results can contain Modules, which report(testset) cannot deepcopy.
    flattened = TestReports.flatten_results!(results)
    write_junit(report(flattened), output)
    any_problems(flattened) && exit(1)
end
