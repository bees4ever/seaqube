from os.path import join

if __name__ == "__main__":
    runner = ModelTrainingBenchmark()

    # first train a model for later

    #model_manager = FTModelStd500V5()
    dataset = "imdb"
    small_ds = join(config['data_storage_pipelines_base_url'], "datasets", dataset, "small",
                          f"{dataset}_small.json")

    model_path = join(config['data_storage_pipelines_base_url'], "models", dataset, "small_imdb_model01.pck")

    #model_manager.process(load_json(small_ds))
    #model_manager.get()

    #dill_dumper(model_manager.get(), model_path)


    weasel_model = CustomNLPLoader.load_model_from_path(model_path)

    
    #anna = WordAnalogyBenchmark("google-analogies")
    #print(anna(weasel_model.model))


    for name in ['semeval', 'jair', 'sat', 'msr']:
        anna = WordAnalogyBenchmark(name)
        print(name, anna(weasel_model.model))


    #runner("imdb", 3, FTModelStd500V5, [WordSimilarityBenchmark('simlex999'), WordSimilarityBenchmark('mturk-771'), anna])
