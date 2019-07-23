import Stable_Marriage
import Stable_Marriage_Syntax
import os
from multiprocessing import Process, Pool


def main():
 pool = Pool(processes=2)
 procs=list()
 basedir = "C:/Users/D072202/DeepAnyMatch/DeepAnyMatch/result_data/oaei/"#OAEI_w2v_steps_walklength1_3grams_2019_06_17_23_44_39_385507/"
 for name in os.listdir(basedir):
   root = os.path.join(basedir,name)
   if os.path.isdir(root):
      #Stable_Marriage.match(os.path.join(root, name))
      #Stable_Marriage_Syntax.match(os.path.join(root, name))proc = Process(target=doubler, args=(number,))
      #proc = Process(target=Stable_Marriage.match, args=(root+str(os.sep),))
      pool.apply_async(Stable_Marriage.match, args=(root+str(os.sep),))
      #procs.append(proc)
      #proc.start()
      #proc = Process(target=Stable_Marriage_Syntax.match, args=(root+str(os.sep),))
      pool.apply_async(Stable_Marriage_Syntax.match, args=(root+str(os.sep),))
      #procs.append(proc)
      #proc.start()
      print("Started " + name)
 
 #for proc in procs:
   #proc.join()
 print("Done")