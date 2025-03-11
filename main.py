from utils import init_args
from runnable import Runnable

if __name__ == "__main__":
    args = init_args()
    runner = Runnable(args)

    # Embed mode
    if args["embed"] and not args["query"]:
        print("Running in EMBED mode.")
        runner.run()  
        print("Embedding completed successfully. Exiting.")
    
    # Query mode
    if args["query"] and not args["embed"]:
        print("Running in QUERY mode.")

        if args["use_llama"]:
            results = runner.run_with_llama()
        else:
            results = runner.run()

        # parsing of results if ensemble is not used
        if not args["use_ensemble"]:
            if results is None:
                print("No results found in the vector store.")
                results = []
            else:
                results = [r[0] for r in results]

        # LLM extraction
        result_llm = runner.run_value_extraction(results)
        print(result_llm)

    # Check mutually exclusive parameters
    if args["query"] and args["embed"]:
        print("Error: You must specify either --embed or --query.")

