import os
package_directory = os.path.dirname(os.path.abspath(__file__))
from cle.configurations.PipelineTools import PipelineDataTuple
from cle.configurations.DiskDataPreparation import prepare_file, prepare_dir, clean_cache
import time

class ConfigurationHandler:

    def execute(self, configuration):

        #try:

            clean_cache(configuration.cachedir)
            prepare_dir(configuration.rundir)
            prepare_dir(configuration.cachedir)
            prepare_file(configuration.logfile)

            #open(configuration.logfile,"w+").close()

            current_time_in_millis = int(round(time.time()))


            print("-----------------------------------------------")
            print("Starting '" + configuration.name + "':\n\n")


            configuration.logs_ = open(configuration.logfile, "a+")
            configuration.log(configuration.to_string())

            prepare_file(configuration.src_corpus)
            prepare_file(configuration.tgt_corpus)
            prepare_file(configuration.src_triples)
            prepare_file(configuration.tgt_triples)
            for path_to_file in configuration.gold_mapping.raw_trainsets + configuration.gold_mapping.raw_testsets:
                prepare_file(path_to_file)



            step = configuration.pipeline.get_first_step()
            while step is not None:
                print("Performing step " + str(step.func.__module__) + "." + str(step.func.__name__) + " with " + str(step.args))
                x = []
                if step.input_step is not None:
                    for input_step in step.input_step.elems:
                        for elem in input_step.output.elems:
                            x.append(elem)
                t = PipelineDataTuple(*x)
                out = step.func(t, step.args, configuration)
                if step.persist_output:
                    step.output = out

                step = step.next_step


            configuration.log("\n\n\n\nNeeded " + str(int(round(time.time())) - current_time_in_millis) + "s.")
            configuration.logs_.close()

            print("-----------------------------------------------")
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
        #    print(configuration.name + " FAILED.")
        #    print("-----------------------------------------------")

            del configuration

