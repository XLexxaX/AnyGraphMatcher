import os
package_directory = os.path.dirname(os.path.abspath(__file__))
from configurations.PipelineTools import PipelineDataTuple
from configurations.DiskDataPreparation import prepare_file, prepare_dir, clean_cache
import time

class ConfigurationHandler:

    def execute(self, configuration):

        #try:

            clean_cache(configuration.cachedir)
            prepare_dir(configuration.resultsdir)
            prepare_dir(configuration.rundir)
            prepare_dir(configuration.cachedir)
            prepare_file(configuration.logfile)

            #open(configuration.logfile,"w+").close()

            current_time_in_millis = int(round(time.time()))


            configuration.logs_ = open(configuration.logfile, "a+", encoding="UTF-8")
            configuration.log("Starting '" + configuration.name + "'\n")

            configuration.log("Configuration as follows:")
            configuration.log(configuration.to_string())

            configuration.log("\nExecution as follows:")

            #prepare_file(configuration.src_corpus)
            #prepare_file(configuration.tgt_corpus)
            prepare_file(configuration.src_triples)
            prepare_file(configuration.tgt_triples)
            for path_to_file in configuration.gold_mapping.raw_trainsets + configuration.gold_mapping.raw_testsets:
                prepare_file(path_to_file)



            step = configuration.pipeline.get_first_step()
            while step is not None:
                configuration.log("Performing step " + str(step.func.__module__) + "." + str(step.func.__name__) + " with " + str(step.args))
                x = []
                if step.input_step is not None:
                    for input_step in step.input_step.elems:
                        for elem in input_step.output.elems:
                            x.append(elem)

                #clean memory
                step.input_step = None

                t = PipelineDataTuple(*x)
                out = step.func(t, step.args, configuration)

                if step.persist_output:
                    step.output = out

                step = step.next_step


            configuration.log("\n\nSuccessfully finished '" + configuration.name + "'.")
            configuration.log("Needed " + str(int(round(time.time())) - current_time_in_millis) + "s.")

            configuration.log("-----------------------------------------------")
            configuration.logs_.close()
        #except Exception as e:
        #    try:
        #        configuration.logs_.close()
        #    except:
        #        pass
        #    logs = open(configuration.logfile, "a+")
        #    logs.write("\n\n\n")
        #    logs.write(configuration.name + " FAILED due to:")
        #    logs.write(str(e))
        #    logs.close()
        #    CONFIGURATION.log(configuration.name + " FAILED.")
        #    CONFIGURATION.log("-----------------------------------------------")

            del configuration
