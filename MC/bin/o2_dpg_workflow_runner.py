#!/usr/bin/env python3

# started February 2021, sandro.wenzel@cern.ch

import re
import subprocess
import shlex
import time
import json
import logging
import os
import signal
import socket
import sys
import traceback
import platform
import numpy as np
try:
    from graphviz import Digraph
    havegraphviz=True
except ImportError:
    havegraphviz=False

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

sys.setrecursionlimit(100000)

import argparse
import psutil
max_system_mem=psutil.virtual_memory().total

sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'o2dpg_workflow_utils'))
from o2dpg_workflow_utils import read_workflow

# defining command line options
parser = argparse.ArgumentParser(description='Parallel execution of a (O2-DPG) DAG data/job pipeline under resource contraints.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-f','--workflowfile', help='Input workflow file name', required=True)
parser.add_argument('-jmax','--maxjobs', help='Number of maximal parallel tasks.', default=100)
parser.add_argument('-k','--keep-going', action='store_true', help='Keep executing the pipeline as far possibe (not stopping on first failure)')
parser.add_argument('--dry-run', action='store_true', help='Show what you would do.')
parser.add_argument('--visualize-workflow', action='store_true', help='Saves a graph visualization of workflow.')
parser.add_argument('--target-labels', nargs='+', help='Runs the pipeline by target labels (example "TPC" or "DIGI").\
                    This condition is used as logical AND together with --target-tasks.', default=[])
parser.add_argument('-tt','--target-tasks', nargs='+', help='Runs the pipeline by target tasks (example "tpcdigi"). By default everything in the graph is run. Regular expressions supported.', default=["*"])
parser.add_argument('--produce-script', help='Produces a shell script that runs the workflow in serialized manner and quits.')
parser.add_argument('--rerun-from', help='Reruns the workflow starting from given task (or pattern). All dependent jobs will be rerun.')
parser.add_argument('--list-tasks', help='Simply list all tasks by name and quit.', action='store_true')
parser.add_argument('--update-resources', dest="update_resources", help='Read resource estimates from a JSON and apply where possible.')

parser.add_argument('--mem-limit', help='Set memory limit as scheduling constraint (in MB)', default=0.9*max_system_mem/1024./1024)
parser.add_argument('--cpu-limit', help='Set CPU limit (core count)', default=8)
parser.add_argument('--cgroup', help='Execute pipeline under a given cgroup (e.g., 8coregrid) emulating resource constraints. This m\
ust exist and the tasks file must be writable to with the current user.')
parser.add_argument('--stdout-on-failure', action='store_true', help='Print log files of failing tasks to stdout,')
parser.add_argument('--webhook', help=argparse.SUPPRESS) # log some infos to this webhook channel
parser.add_argument('--checkpoint-on-failure', help=argparse.SUPPRESS) # debug option making a debug-tarball and sending to specified address
                                                                       # argument is alien-path
parser.add_argument('--retry-on-failure', help=argparse.SUPPRESS, default=0) # number of times a failing task is retried
parser.add_argument('--no-rootinit-speedup', help=argparse.SUPPRESS, action='store_true') # disable init of ROOT environment vars to speedup init/startup
# options to control/boost performance
parser.add_argument("--optimise-cpu", dest="optimise_cpu", action="store_true", help=argparse.SUPPRESS) # try to optimise CPU efficiency during scheduling
parser.add_argument("--dynamic-resources", dest="dynamic_resources", action="store_true", help=argparse.SUPPRESS) # derive resources dynamically

parser.add_argument('--action-logfile', help='Logfilename for action logs. If none given, pipeline_action_#PID.log will be used')
parser.add_argument('--metric-logfile', help='Logfilename for metric logs. If none given, pipeline_metric_#PID.log will be used')
args = parser.parse_args()

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# first file logger
actionlogger = setup_logger('pipeline_action_logger', ('pipeline_action_' + str(os.getpid()) + '.log', args.action_logfile)[args.action_logfile!=None], level=logging.DEBUG)

# second file logger
metriclogger = setup_logger('pipeline_metric_logger', ('pipeline_metric_' + str(os.getpid()) + '.log', args.action_logfile)[args.action_logfile!=None])

# Immediately log imposed memory and CPU limit as well as further useful meta info
_ , meta = read_workflow(args.workflowfile)
meta["cpu_limit"] = args.cpu_limit
meta["mem_limit"] = args.mem_limit
meta["workflow_file"] = os.path.abspath(args.workflowfile)
meta["target_task"] = args.target_tasks
meta["rerun_from"] = args.rerun_from
meta["target_labels"] = args.target_labels
metriclogger.info(meta)

# for debugging without terminal access
# TODO: integrate into standard logger
def send_webhook(hook, t):
    if hook!=None:
      command="curl -X POST -H 'Content-type: application/json' --data '{\"text\":\" " + str(t) + "\"}' " + str(hook) + " &> /dev/null"
      os.system(command)

# A fallback solution to getting all child procs
# in case psutil has problems (PermissionError).
# It returns the same list as psutil.children(recursive=True).
def getChildProcs(basepid):
  cmd='''
  childprocs() {
  local parent=$1
  if [ ! "$2" ]; then
    child_pid_list=""
  fi
  if [ "$parent" ] ; then
    child_pid_list="$child_pid_list $parent"
    for childpid in $(pgrep -P ${parent}); do
      childprocs $childpid "nottoplevel"
    done;
  fi
  # return via a string list (only if toplevel)
  if [ ! "$2" ]; then
    echo "${child_pid_list}"
  fi
  }
  '''
  cmd = cmd + '\n' + 'childprocs ' + str(basepid)
  output = subprocess.check_output(cmd, shell=True)
  plist = []
  for p in output.strip().split():
     try:
         proc=psutil.Process(int(p))
     except psutil.NoSuchProcess:
         continue

     plist.append(proc)
  return plist

#
# Code section to find all topological orderings
# of a DAG. This is used to know when we can schedule
# things in parallel.
# Taken from https://www.geeksforgeeks.org/all-topological-sorts-of-a-directed-acyclic-graph/

# class to represent a graph object
class Graph:

    # Constructor
    def __init__(self, edges, N):

        # A List of Lists to represent an adjacency list
        self.adjList = [[] for _ in range(N)]

        # stores in-degree of a vertex
        # initialize in-degree of each vertex by 0
        self.indegree = [0] * N

        # add edges to the undirected graph
        for (src, dest) in edges:

            # add an edge from source to destination
            self.adjList[src].append(dest)

            # increment in-degree of destination vertex by 1
            self.indegree[dest] = self.indegree[dest] + 1

# Recursive function to find all topological orderings of a given DAG
def findAllTopologicalOrders(graph, path, discovered, N, allpaths, maxnumber=1):
    if len(allpaths) >= maxnumber:
        return

    # do for every vertex
    for v in range(N):

        # proceed only if in-degree of current node is 0 and
        # current node is not processed yet
        if graph.indegree[v] == 0 and not discovered[v]:

            # for every adjacent vertex u of v, reduce in-degree of u by 1
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] - 1

            # include current node in the path and mark it as discovered
            path.append(v)
            discovered[v] = True

            # recur
            findAllTopologicalOrders(graph, path, discovered, N, allpaths)

            # backtrack: reset in-degree information for the current node
            for u in graph.adjList[v]:
                graph.indegree[u] = graph.indegree[u] + 1

            # backtrack: remove current node from the path and
            # mark it as undiscovered
            path.pop()
            discovered[v] = False

    # record valid ordering
    if len(path) == N:
        allpaths.append(path.copy())


# get all topological orderings of a given DAG as a list
def printAllTopologicalOrders(graph, maxnumber=1):
    # get number of nodes in the graph
    N = len(graph.adjList)

    # create an auxiliary space to keep track of whether vertex is discovered
    discovered = [False] * N

    # list to store the topological order
    path = []
    allpaths = []
    # find all topological ordering and print them
    findAllTopologicalOrders(graph, path, discovered, N, allpaths, maxnumber=maxnumber)
    return allpaths

# <--- end code section for topological sorts

# find all tasks that depend on a given task (id); when a cache
# dict is given we can fill for the whole graph in one pass...
def find_all_dependent_tasks(possiblenexttask, tid, cache=None):
    c=cache.get(tid) if cache else None
    if c!=None:
        return c

    daughterlist=[tid]
    # possibly recurse
    for n in possiblenexttask[tid]:
        c = cache.get(n) if cache else None
        if c == None:
            c = find_all_dependent_tasks(possiblenexttask, n, cache)
        daughterlist = daughterlist + c
        if cache is not None:
            cache[n]=c

    if cache is not None:
        cache[tid]=daughterlist
    return list(set(daughterlist))


# wrapper taking some edges, constructing the graph,
# obtain all topological orderings and some other helper data structures
def analyseGraph(edges, nodes):
    # Number of nodes in the graph
    N = len(nodes)

    # candidate list trivial
    nextjobtrivial = { n:[] for n in nodes }
    # startnodes
    nextjobtrivial[-1] = nodes
    for e in edges:
        nextjobtrivial[e[0]].append(e[1])
        if nextjobtrivial[-1].count(e[1]):
            nextjobtrivial[-1].remove(e[1])

    # find topological orderings of the graph
    # create a graph from edges
    graph = Graph(edges, N)
    orderings = printAllTopologicalOrders(graph)

    return (orderings, nextjobtrivial)


def draw_workflow(workflowspec):
    if not havegraphviz:
        print('graphviz not installed, cannot draw workflow')
        return

    dot = Digraph(comment='MC workflow')
    nametoindex={}
    index=0
    # nodes
    for node in workflowspec['stages']:
        name=node['name']
        nametoindex[name]=index
        dot.node(str(index), name)
        index=index+1

    # edges
    for node in workflowspec['stages']:
        toindex = nametoindex[node['name']]
        for req in node['needs']:
            fromindex = nametoindex[req]
            dot.edge(str(fromindex), str(toindex))

    dot.render('workflow.gv')

# builds the graph given a "taskuniverse" list
# builds accompagnying structures tasktoid and idtotask
def build_graph(taskuniverse, workflowspec):
    tasktoid={ t[0]['name']:i for i, t in enumerate(taskuniverse, 0) }
    # print (tasktoid)

    nodes = []
    edges = []
    for t in taskuniverse:
        nodes.append(tasktoid[t[0]['name']])
        for n in t[0]['needs']:
            edges.append((tasktoid[n], tasktoid[t[0]['name']]))

    return (edges, nodes)


# loads json into dict, e.g. for workflow specification
def load_json(workflowfile):
    fp=open(workflowfile)
    workflowspec=json.load(fp)
    return workflowspec


# filters the original workflowspec according to wanted targets or labels
# returns a new workflowspec
def filter_workflow(workflowspec, targets=[], targetlabels=[]):
    if len(targets)==0:
       return workflowspec
    if len(targetlabels)==0 and len(targets)==1 and targets[0]=="*":
       return workflowspec

    transformedworkflowspec = workflowspec

    def task_matches(t):
        for filt in targets:
            if filt=="*":
                return True
            if re.match(filt, t)!=None:
                return True
        return False

    def task_matches_labels(t):
        # when no labels are given at all it's ok
        if len(targetlabels)==0:
            return True

        for l in t['labels']:
            if targetlabels.count(l)!=0:
                return True
        return False

    # The following sequence of operations works and is somewhat structured.
    # However, it builds lookups used elsewhere as well, so some CPU might be saved by reusing
    # some structures across functions or by doing less passes on the data.

    # helper lookup
    tasknametoid = { t['name']:i for i, t in enumerate(workflowspec['stages'],0) }

    # check if a task can be run at all
    # or not due to missing requirements
    def canBeDone(t,cache={}):
       ok = True
       c = cache.get(t['name'])
       if c != None:
           return c
       for r in t['needs']:
           taskid = tasknametoid.get(r)
           if taskid != None:
             if not canBeDone(workflowspec['stages'][taskid], cache):
                ok = False
                break
           else:
             ok = False
             break
       cache[t['name']] = ok
       return ok

    okcache = {}
    # build full target list
    full_target_list = [ t for t in workflowspec['stages'] if task_matches(t['name']) and task_matches_labels(t) and canBeDone(t,okcache) ]
    full_target_name_list = [ t['name'] for t in full_target_list ]

    # build full dependency list for a task t
    def getallrequirements(t):
        _l=[]
        for r in t['needs']:
            fulltask = workflowspec['stages'][tasknametoid[r]]
            _l.append(fulltask)
            _l=_l+getallrequirements(fulltask)
        return _l

    full_requirements_list = [ getallrequirements(t) for t in full_target_list ]

    # make flat and fetch names only
    full_requirements_name_list = list(set([ item['name'] for sublist in full_requirements_list for item in sublist ]))

    # inner "lambda" helper answering if a task "name" is needed by given targets
    def needed_by_targets(name):
        if full_target_name_list.count(name)!=0:
            return True
        if full_requirements_name_list.count(name)!=0:
            return True
        return False

    # we finaly copy everything matching the targets as well
    # as all their requirements
    transformedworkflowspec['stages']=[ l for l in workflowspec['stages'] if needed_by_targets(l['name']) ]
    return transformedworkflowspec


# builds topological orderings (for each timeframe)
def build_dag_properties(workflowspec):
    globaltaskuniverse = [ (l, i) for i, l in enumerate(workflowspec['stages'], 1) ]
    timeframeset = set( l['timeframe'] for l in workflowspec['stages'] )

    edges, nodes = build_graph(globaltaskuniverse, workflowspec)
    tup = analyseGraph(edges, nodes.copy())
    #
    global_next_tasks = tup[1]


    dependency_cache = {}
    # weight influences scheduling order can be anything user defined ... for the moment we just prefer to stay within a timeframe
    # then take the number of tasks that depend on a task as further weight
    # TODO: bring in resource estimates from runtime, CPU, MEM
    # TODO: make this a policy of the runner to study different strategies
    def getweight(tid):
        return (globaltaskuniverse[tid][0]['timeframe'], len(find_all_dependent_tasks(global_next_tasks, tid, dependency_cache)))

    task_weights = [ getweight(tid) for tid in range(len(globaltaskuniverse)) ]

    for tid in range(len(globaltaskuniverse)):
        actionlogger.info("Score for " + str(globaltaskuniverse[tid][0]['name']) + " is " + str(task_weights[tid]))

    # print (global_next_tasks)
    return { 'nexttasks' : global_next_tasks, 'weights' : task_weights, 'topological_ordering' : tup[0] }

# update the resource estimates of a workflow based on resources given via JSON
def update_resource_estimates(workflow, resource_json):
    resource_dict = load_json(resource_json)
    stages = workflow["stages"]

    for task in stages:
        if task["timeframe"] >= 1:
            tf = task["timeframe"]
            name = "_".join(task["name"].split("_")[:-1])
        else:
            name = task["name"]

        if name not in resource_dict:
            continue

        new_resources = resource_dict[name]

        # memory
        newmem = new_resources.get("mem", None)
        if newmem is not None:
            oldmem = task["resources"]["mem"]
            actionlogger.info("Updating mem estimate for " + task["name"] + " from " + str(oldmem) + " to " + str(newmem))
            task["resources"]["mem"] = newmem
        newcpu = new_resources.get("cpu", None)

        # cpu
        if newcpu is not None:
            oldcpu = task["resources"]["cpu"]
            rel_cpu = task["resources"]["relative_cpu"]
            if rel_cpu is not None:
               # respect the relative CPU settings
               # By default, the CPU value in the workflow is already scaled if relative_cpu is given.
               # The new estimate on the other hand is not yet scaled so it needs to be done here.
               newcpu *= rel_cpu
            actionlogger.info("Updating cpu estimate for " + task["name"] + " from " + str(oldcpu) + " to " + str(newcpu))
            task["resources"]["cpu"] = newcpu

# a function to read a software environment determined by alienv into
# a python dictionary
def get_alienv_software_environment(packagestring):
    """
    packagestring is something like O2::v202298081-1,O2Physics::xxx
    """
    # alienv printenv packagestring --> dictionary
    # for the moment this works with CVMFS only
    cmd="/cvmfs/alice.cern.ch/bin/alienv printenv " + packagestring
    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    envstring, err = proc.communicate()
    # see if the printenv command was successful
    if len(err.decode()) > 0:
       print (err.decode())
       raise Exception

    # the software environment is now in the evnstring
    # split it on semicolon
    envstring=envstring.decode()
    tokens=envstring.split(";")
    # build envmap
    envmap = {}
    for t in tokens:
      # check if assignment
      if t.count("=") > 0:
         assignment = t.rstrip().split("=")
         envmap[assignment[0]] = assignment[1]
      elif t.count("export") > 0:
         # the case when we export or a simple variable
         # need to consider the case when this has not been previously assigned
         variable = t.split()[1]
         if not variable in envmap:
            envmap[variable]=""

    return envmap

#
# functions for execution; encapsulated in a WorkflowExecutor class
#
class WorkflowExecutor:
    # Constructor
    def __init__(self, workflowfile, args, jmax=100):
      self.args=args
      self.workflowfile = workflowfile
      self.workflowspec = load_json(workflowfile)
      self.globalenv = self.extract_global_environment(self.workflowspec) # initialize global environment settings
      for e in self.globalenv:
        if os.environ.get(e, None) == None:
           actionlogger.info("Applying global environment from init section " + str(e) + " : " + str(self.globalenv[e]))
           os.environ[e] = str(self.globalenv[e])

      self.workflowspec = filter_workflow(self.workflowspec, args.target_tasks, args.target_labels)

      if not self.workflowspec['stages']:
          if args.target_tasks:
              print ('Apparently some of the chosen target tasks are not in the workflow')
              exit (0)
          print ('Workflow is empty. Nothing to do')
          exit (0)

      workflow = build_dag_properties(self.workflowspec)
      if args.visualize_workflow:
          draw_workflow(self.workflowspec)
      self.possiblenexttask = workflow['nexttasks']
      self.taskweights = workflow['weights']
      self.topological_orderings = workflow['topological_ordering']
      self.taskuniverse = [ l['name'] for l in self.workflowspec['stages'] ]
      self.idtotask = [ 0 for l in self.taskuniverse ]
      self.tasktoid = {}
      for i in range(len(self.taskuniverse)):
          self.tasktoid[self.taskuniverse[i]]=i
          self.idtotask[i]=self.taskuniverse[i]

      if args.update_resources:
          update_resource_estimates(self.workflowspec, args.update_resources)

      self.maxmemperid = [ float(self.workflowspec['stages'][tid]['resources']['mem']) for tid in range(len(self.taskuniverse)) ]
      self.cpuperid = [ float(self.workflowspec['stages'][tid]['resources']['cpu']) for tid in range(len(self.taskuniverse)) ]
      self.maxmemperid_persistent = self.maxmemperid.copy()
      self.cpuperid_persistent = self.cpuperid.copy()
      self.resources_running_dict = {}
      self.resources_running_tid = []
      for _, task in enumerate(self.workflowspec['stages']):
        global_task_name = self.get_global_task_name(task["name"])
        if global_task_name not in self.resources_running_dict:
            self.resources_running_dict[global_task_name] = [False, None, None, [], task["resources"]["relative_cpu"] if task["resources"]["relative_cpu"] else 1]
        # now we can look everything up via tid
        self.resources_running_tid.append(self.resources_running_dict[global_task_name])

      self.curmembooked = 0
      self.curcpubooked = 0
      self.curmembooked_backfill = 0
      self.curcpubooked_backfill = 0
      self.memlimit = float(args.mem_limit) # some configurable number
      self.cpulimit = float(args.cpu_limit)
      self.procstatus = { tid:'ToDo' for tid in range(len(self.workflowspec['stages'])) }
      self.taskneeds= { t:set(self.getallrequirements(t)) for t in self.taskuniverse }
      self.stoponfailure = not (args.keep_going == True)
      print ("Stop on failure ",self.stoponfailure)
      self.max_jobs_parallel = int(jmax)
      self.scheduling_iteration = 0
      self.process_list = []  # list of currently scheduled tasks with normal priority
      self.backfill_process_list = [] # list of curently scheduled tasks with low backfill priority (not sure this is needed)
      self.pid_to_psutilsproc = {}  # cache of putilsproc for resource monitoring
      self.pid_to_files = {} # we can auto-detect what files are produced by which task (at least to some extent)
      self.pid_to_connections = {} # we can auto-detect what connections are opened by which task (at least to some extent)
      signal.signal(signal.SIGINT, self.SIGHandler)
      signal.siginterrupt(signal.SIGINT, False)
      self.nicevalues = [ os.nice(0) for tid in range(len(self.taskuniverse)) ]
      self.internalmonitorcounter = 0 # internal use
      self.internalmonitorid = 0 # internal use
      self.tids_marked_toretry = [] # sometimes we might want to retry a failed task (simply because it was "unlucky") and we put them here
      self.retry_counter = [ 0 for tid in range(len(self.taskuniverse)) ] # we keep track of many times retried already
      self.task_retries = [ self.workflowspec['stages'][tid].get('retry_count',0) for tid in range(len(self.taskuniverse)) ] # the per task specific "retry" number -> needs to be parsed from the JSON

      self.semaphore_values = { self.workflowspec['stages'][tid].get('semaphore'):0 for tid in range(len(self.taskuniverse)) if self.workflowspec['stages'][tid].get('semaphore')!=None } # keeps current count of semaphores (defined in the json workflow). used to achieve user-defined "critical sections".
      self.alternative_envs = {} # mapping of taskid to alternative software envs (to be applied on a per-task level)
      # init alternative software environments
      self.init_alternative_software_environments()

      self.try_job_from_candidates_backfill_impl = self.try_job_from_candidates_backfill_default
      if args.optimise_cpu:
        self.try_job_from_candidates_impl = self.try_job_from_candidates_cpu
      else:
        self.try_job_from_candidates_impl = self.try_job_from_candidates_default

    def SIGHandler(self, signum, frame):
       # basically forcing shut down of all child processes
       actionlogger.info("Signal " + str(signum) + " caught")
       try:
           procs = psutil.Process().children(recursive=True)
       except (psutil.NoSuchProcess):
           pass
       except (psutil.AccessDenied, PermissionError):
           procs = getChildProcs(os.getpid())

       for p in procs:
           actionlogger.info("Terminating " + str(p))
           try:
             p.terminate()
           except (psutil.NoSuchProcess, psutil.AccessDenied):
             pass

       gone, alive = psutil.wait_procs(procs, timeout=3)
       for p in alive:
           try:
             actionlogger.info("Killing " + str(p))
             p.kill()
           except (psutil.NoSuchProcess, psutil.AccessDenied):
             pass

       exit (1)


    def extract_global_environment(self, workflowspec):
        """Checks if the workflow contains a dedicated init task
           defining a global environment. Extract information and remove from workflowspec.
        """
        init_index = 0 # this has to be the first task in the workflow
        globalenv = {}
        if workflowspec['stages'][init_index]['name'] == '__global_init_task__':
          env = workflowspec['stages'][init_index].get('env', None)
          if env != None:
            globalenv = { e : env[e] for e in env }
          del workflowspec['stages'][init_index]

        return globalenv

    def getallrequirements(self, t):
        l=[]
        for r in self.workflowspec['stages'][self.tasktoid[t]]['needs']:
            l.append(r)
            l=l+self.getallrequirements(r)
        return l

    def get_done_filename(self, tid):
        name = self.workflowspec['stages'][tid]['name']
        workdir = self.workflowspec['stages'][tid]['cwd']
        # name and workdir define the "done" file as used by taskwrapper
        # this assumes that taskwrapper is used to actually check if something is to be rerun
        done_filename = workdir + '/' + name + '.log_done'
        return done_filename

    def get_res_filename(self, tid):
        name = self.workflowspec['stages'][tid]['name']
        workdir = self.workflowspec['stages'][tid]['cwd']
        # name and workdir define the "done" file as used by taskwrapper
        # this assumes that taskwrapper is used to actually check if something is to be rerun
        return os.path.join(workdir, f"{name}.log_time")

    # removes the done flag from tasks that need to be run again
    def remove_done_flag(self, listoftaskids):
       for tid in listoftaskids:
          done_filename = self.get_done_filename(tid)
          name=self.workflowspec['stages'][tid]['name']
          if args.dry_run:
              print ("Would mark task " + name + " as to be done again")
          else:
              print ("Marking task " + name + " as to be done again")
              if os.path.exists(done_filename) and os.path.isfile(done_filename):
                  os.remove(done_filename)

    # submits a task as subprocess and records Popen instance
    def submit(self, tid, nice=os.nice(0)):
      actionlogger.debug("Submitting task " + str(self.idtotask[tid]) + " with nice value " + str(nice))
      c = self.workflowspec['stages'][tid]['cmd']
      workdir = self.workflowspec['stages'][tid]['cwd']
      if not workdir=='':
          if os.path.exists(workdir) and not os.path.isdir(workdir):
                  actionlogger.error('Cannot create working dir ... some other resource exists already')
                  return None

          if not os.path.isdir(workdir):
                  os.makedirs(workdir)

      self.procstatus[tid]='Running'
      if args.dry_run:
          drycommand="echo \' " + str(self.scheduling_iteration) + " : would do " + str(self.workflowspec['stages'][tid]['name']) + "\'"
          return subprocess.Popen(['/bin/bash','-c',drycommand], cwd=workdir)

      taskenv = os.environ.copy()
      # add task specific environment
      if self.workflowspec['stages'][tid].get('env')!=None:
          taskenv.update(self.workflowspec['stages'][tid]['env'])

      # apply specific (non-default) software version, if any
      # (this was setup earlier)
      alternative_env = self.alternative_envs.get(tid, None)
      if alternative_env != None:
          actionlogger.info('Applying alternative software environment to task ' + self.idtotask[tid])
          for entry in alternative_env:
              # overwrite what is present in default
              taskenv[entry] = alternative_env[entry]

      p = psutil.Popen(['/bin/bash','-c',c], cwd=workdir, env=taskenv)
      try:
          p.nice(nice)
          self.nicevalues[tid]=nice
      except (psutil.NoSuchProcess, psutil.AccessDenied):
          actionlogger.error('Couldn\'t set nice value of ' + str(p.pid) + ' to ' + str(nice))
          self.nicevalues[tid]=os.nice(0)
      return p

    def ok_to_submit(self, tid, backfill=False):
      softcpufactor=1
      softmemfactor=1
      if backfill:
          softcpufactor=1.5
          sotmemfactor=1.5

      # check semaphore
      sem = self.workflowspec['stages'][tid].get('semaphore')
      if sem != None:
        if self.semaphore_values[sem] > 0:
           return False

      maxcpu = self.cpuperid[tid]
      maxmem = self.maxmemperid[tid]
      actionlogger.info("Setup resources for task %s, cpu: %d, mem: %d", self.idtotask[tid], maxcpu, maxmem)
      # check other resources
      if not backfill:
          # analyse CPU
          okcpu = (self.curcpubooked + maxcpu <= self.cpulimit)
          # analyse MEM
          okmem = (self.curmembooked + maxmem <= self.memlimit)
          actionlogger.debug ('Condition check --normal-- for  ' + str(tid) + ':' + str(self.idtotask[tid]) + ' CPU ' + str(okcpu) + ' MEM ' + str(okmem))
          return (okcpu and okmem)
      else:
        # only backfill one job at a time
        if self.curcpubooked_backfill > 0:
            return False

        # not backfilling jobs which either take much memory or use lot's of CPU anyway
        # conditions are somewhat arbitrary and can be played with
        if maxcpu > 0.9*float(self.args.cpu_limit):
            return False
        if maxmem/float(self.args.cpu_limit) >= 1900:
            return False

        # analyse CPU
        okcpu = (self.curcpubooked_backfill + maxcpu <= self.cpulimit)
        okcpu = okcpu and (self.curcpubooked + self.curcpubooked_backfill + maxcpu <= softcpufactor*self.cpulimit)
        # analyse MEM
        okmem = (self.curmembooked + self.curmembooked_backfill + maxmem <= softmemfactor*self.memlimit)
        actionlogger.debug ('Condition check --backfill-- for  ' + str(tid) + ':' + str(self.idtotask[tid]) + ' CPU ' + str(okcpu) + ' MEM ' + str(okmem))
        return (okcpu and okmem)
      return False


    def ok_to_skip(self, tid):
        done_filename = self.get_done_filename(tid)
        if os.path.exists(done_filename) and os.path.isfile(done_filename):
            return True
        return False

    def book_resources(self, tid, backfill = False):
        # books the resources used by a certain task
        # semaphores
        sem = self.workflowspec['stages'][tid].get('semaphore')
        if sem != None:
          self.semaphore_values[sem]+=1

        # CPU + MEM
        if not backfill:
          self.curmembooked+=self.maxmemperid[tid]
          self.curcpubooked+=self.cpuperid[tid]
        else:
          self.curmembooked_backfill+=self.maxmemperid[tid]
          self.curcpubooked_backfill+=self.cpuperid[tid]

    def unbook_resources(self, tid, backfill = False):
        # "frees" the nominal resources used by a certain task from the accounting
        # so that other jobs can be scheduled
        sem = self.workflowspec['stages'][tid].get('semaphore')
        if sem != None:
          self.semaphore_values[sem]-=1

        # CPU + MEM
        if not backfill:
          self.curmembooked-=self.maxmemperid[tid]
          self.curcpubooked-=self.cpuperid[tid]
        else:
          self.curmembooked_backfill-=self.maxmemperid[tid]
          self.curcpubooked_backfill-=self.cpuperid[tid]


    def try_job_from_candidates_default(self, taskcandidates):
        for tid in taskcandidates.copy():
            actionlogger.debug ("trying to submit " + str(tid) + ':' + str(self.idtotask[tid]))
            self.update_resources(tid)
            if (len(self.process_list) + len(self.backfill_process_list) < self.max_jobs_parallel) and self.ok_to_submit(tid):
                p=self.submit(tid)
                if p!=None:
                    self.book_resources(tid)
                    self.process_list.append((tid,p))
                    taskcandidates.remove(tid)
                    # minimal delay
                    time.sleep(0.1)
            else:
                break #---> we break at first failure assuming some priority (other jobs may come in via backfill)


    def try_job_from_candidates_cpu(self, taskcandidates):
        # collect all that would be ok to submit
        possible_submit = []
        # the cpu and mem the process will potentially take
        potential_cpu_mem_time = []

        for tid in taskcandidates.copy():
            actionlogger.debug ("trying to submit " + str(tid) + ':' + str(self.idtotask[tid]))
            self.update_resources(tid)
            if (len(self.process_list) + len(self.backfill_process_list) < self.max_jobs_parallel) and self.ok_to_submit(tid):
                possible_submit.append(tid)
                potential_cpu_mem_time.append((self.cpuperid[tid], self.maxmemperid[tid], self.get_walltime(tid)))

        # see how much resources are free as we go
        residual_cpu_mem_time = [[i, self.cpulimit - self.curcpubooked, self.memlimit - self.curmembooked, []] for i, _ in enumerate(possible_submit)]

        # each list in this list is a combination of task IDs
        build_lists = [[] for _ in possible_submit]

        for i, _ in enumerate(possible_submit):
            # we want to fill each list in build_lists
            for j in range(-len(possible_submit) + i, i):
                this_cpu, this_mem, this_time = potential_cpu_mem_time[j]
                _, res_cpu, res_mem, _ = residual_cpu_mem_time[i]
                if res_cpu - this_cpu < 0 or res_mem - this_mem < 0:
                    # too much mem or cpu would be taken
                    continue
                build_lists[i].append(possible_submit[j])
                residual_cpu_mem_time[i][1] -= this_cpu
                residual_cpu_mem_time[i][2] -= this_mem
                if this_time:
                    # greater 0 and not None
                    residual_cpu_mem_time[i][3].append(this_time)
            # round cpu so that we can also sort meaningfully by another parameter
            residual_cpu_mem_time[i][1] = round(residual_cpu_mem_time[i][1], 1)

        if possible_submit:
            # go for best cpu usage first and after that, sort by number of tasks that could be submitted
            residual_cpu_mem_time.sort(key=lambda tup: (tup[1], -len(build_lists[tup[0]])))
            index = residual_cpu_mem_time[0][0]

            possible_submit = build_lists[index]

        for tid in possible_submit:
            # submit all of these tasks
            p=self.submit(tid)
            if p!=None:
                self.book_resources(tid)
                self.process_list.append((tid,p))
                taskcandidates.remove(tid)
                # minimal delay
                time.sleep(0.1)
            else:
                break #---> we break at first failure assuming some priority (other jobs may come in via backfill)

    def try_job_from_candidates_backfill_default(self, taskcandidates):
        # the backfill part for remaining candidates
        initialcandidates=taskcandidates.copy()

        for tid in initialcandidates:
            actionlogger.debug ("trying to backfill submit " + str(tid) + ':' + str(self.idtotask[tid]))

            if (len(self.process_list) + len(self.backfill_process_list) < self.max_jobs_parallel) and self.ok_to_submit(tid, backfill=True):
                p=self.submit(tid, 19)
                if p!=None:
                    self.book_resources(tid, backfill=True)
                    self.process_list.append((tid,p))
                    taskcandidates.remove(tid) #-> not sure about this one
                    # minimal delay
                    time.sleep(0.1)
            else:
                continue


    def try_job_from_candidates_wrapper(self, taskcandidates, finished):
       self.scheduling_iteration = self.scheduling_iteration + 1

       print(f"Current free CPU: {self.cpulimit - self.curcpubooked}")
       print(f"Current free MEM: {self.memlimit - self.curmembooked}")

       # remove "done / skippable" tasks immediately
       for tid in taskcandidates.copy():  # <--- the copy is important !! otherwise this loop is not doing what you think
          if self.ok_to_skip(tid):
              finished.append(tid)
              taskcandidates.remove(tid)
              actionlogger.info("Skipping task " + str(self.idtotask[tid]))

       self.try_job_from_candidates_impl(taskcandidates)
       self.try_job_from_candidates_backfill_impl(taskcandidates)


    def stop_pipeline_and_exit(self, process_list):
        # kill all remaining jobs
        for p in process_list:
           p[1].kill()

        exit(1)

    def get_global_task_name(self, name):
        tokens = name.split("_")
        try:
            int(tokens[-1])
            return "_".join(tokens[:-1])
        except ValueError:
            pass
        return name


    def add_resources(self, tid):
        """
        Read file that is produced by O2 jobutils2.sh taskwrapper

        In case it doesn't exist, simply return

        Extract CPU, memory and walltime
        """
        filename = self.get_res_filename(tid)
        if not os.path.exists(filename):
            actionlogger.info("DONE filename does not exist at %s", filename)
            return
        res = self.resources_running_tid[tid]
        res[0] = True
        with open(filename, "r") as f:
            for l in f:
                if "CPU" in l:
                    res[1] = max(float(l.strip().split()[-1].split('%')[0]) / 100, res[1] if res[1] else 0)
                elif "mem" in l:
                    res[2] = max(float(l.strip().split()[-1]) / 1024, res[2] if res[2] else 0)
                elif "walltime" in l:
                    res[3].append(float(l.strip().split()[-1]))


    def update_resources(self, tid):
        """
        Upudate the resources from derived ones

        If not possible, take original user estimates
        """
        if not args.dynamic_resources:
            return
        res = self.resources_running_tid[tid]
        self.cpuperid[tid] = min(float(self.cpuperid[tid]) if not res[0] or res[1] is None else res[1] * res[4], self.cpuperid_persistent[tid])
        self.maxmemperid[tid] = min(float(self.maxmemperid[tid]) if not res[0] or res[2] is None else res[2], self.maxmemperid_persistent[tid])

    def get_walltime(self, tid):
        return np.mean(res[3]) if res[0] else None


    def monitor(self, process_list):
        self.internalmonitorcounter+=1
        if self.internalmonitorcounter % 5 != 0:
            return

        self.internalmonitorid+=1



        globalCPU=0.
        globalPSS=0.
        globalCPU_backfill=0.
        globalPSS_backfill=0.
        resources_per_task = {}
        for tid, proc in process_list:
            # proc is Popen object
            pid=proc.pid
            if self.pid_to_files.get(pid)==None:
                self.pid_to_files[pid]=set()
                self.pid_to_connections[pid]=set()
            try:
                psutilProcs = [ proc ]
                # use psutil for CPU measurement
                psutilProcs = psutilProcs + proc.children(recursive=True)
            except (psutil.NoSuchProcess):
                continue

            except (psutil.AccessDenied, PermissionError):
                psutilProcs = psutilProcs + getChildProcs(pid)

            # accumulate total metrics (CPU, memory)
            totalCPU = 0.
            totalPSS = 0.
            totalSWAP = 0.
            totalUSS = 0.
            for p in psutilProcs:
                """
                try:
                    for f in p.open_files():
                        self.pid_to_files[pid].add(str(f.path)+'_'+str(f.mode))
                    for f in p.connections(kind="all"):
                        remote=f.raddr
                        if remote==None:
                            remote='none'
                        self.pid_to_connections[pid].add(str(f.type)+"_"+str(f.laddr)+"_"+str(remote))
                except Exception:
                    pass
                """
                thispss=0
                thisuss=0
                # MEMORY part
                try:
                    fullmem=p.memory_full_info()
                    thispss=getattr(fullmem,'pss',0) #<-- pss not available on MacOS
                    totalPSS=totalPSS + thispss
                    totalSWAP=totalSWAP + fullmem.swap
                    thisuss=fullmem.uss
                    totalUSS=totalUSS + thisuss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                # CPU part
                # fetch existing proc or insert
                cachedproc = self.pid_to_psutilsproc.get(p.pid)
                if cachedproc!=None:
                    try:
                        thiscpu = cachedproc.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        thiscpu = 0.
                    totalCPU = totalCPU + thiscpu
                    # thisresource = {'iter':self.internalmonitorid, 'pid': p.pid, 'cpu':thiscpu, 'uss':thisuss/1024./1024., 'pss':thispss/1024./1024.}
                    # metriclogger.info(thisresource)
                else:
                    self.pid_to_psutilsproc[p.pid] = p
                    try:
                        self.pid_to_psutilsproc[p.pid].cpu_percent()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

            resources_per_task[tid]={'iter':self.internalmonitorid, 'name':self.idtotask[tid], 'cpu':totalCPU, 'uss':totalUSS/1024./1024., 'pss':totalPSS/1024./1024, 'nice':self.nicevalues[tid], 'swap':totalSWAP, 'label':self.workflowspec['stages'][tid]['labels']}

            metriclogger.info(resources_per_task[tid])
            send_webhook(self.args.webhook, resources_per_task)

        for r in resources_per_task.values():
            if r['nice']==os.nice(0):
                globalCPU+=r['cpu']
                globalPSS+=r['pss']
            else:
                globalCPU_backfill+=r['cpu']
                globalPSS_backfill+=r['pss']

        if globalPSS > self.memlimit:
            metriclogger.info('*** MEMORY LIMIT PASSED !! ***')
            # --> We could use this for corrective actions such as killing jobs currently back-filling
            # (or better hibernating)

    def waitforany(self, process_list, finished, failingtasks):
       failuredetected = False
       failingpids = []
       if len(process_list)==0:
           return False

       for p in list(process_list):
          pid = p[1].pid
          tid = p[0]  # the task id of this process
          returncode = 0
          if not self.args.dry_run:
              returncode = p[1].poll()
          if returncode!=None:
            actionlogger.info ('Task ' + str(pid) + ' ' + str(tid)+':'+str(self.idtotask[tid]) + ' finished with status ' + str(returncode))
            # account for cleared resources
            self.unbook_resources(tid, backfill = self.nicevalues[tid]!=os.nice(0) )
            self.procstatus[tid]='Done'
            finished.append(tid)
            self.add_resources(tid)
            #self.validate_resources_running(tid)
            process_list.remove(p)
            if returncode != 0:
               print (str(self.idtotask[tid]) + ' failed ... checking retry')
               # we inspect if this is something "unlucky" which could be resolved by a simple resubmit
               if self.is_worth_retrying(tid) and ((self.retry_counter[tid] < int(args.retry_on_failure)) or (self.retry_counter[tid] < int(self.task_retries[tid]))):
                 print (str(self.idtotask[tid]) + ' to be retried')
                 actionlogger.info ('Task ' + str(self.idtotask[tid]) + ' failed but marked to be retried ')
                 self.tids_marked_toretry.append(tid)
                 self.retry_counter[tid] += 1

               else:
                 failuredetected = True
                 failingpids.append(pid)
                 failingtasks.append(tid)

       if failuredetected and self.stoponfailure:
          actionlogger.info('Stoping pipeline due to failure in stages with PID ' + str(failingpids))
          # self.analyse_files_and_connections()
          if self.args.stdout_on_failure:
             self.cat_logfiles_tostdout(failingtasks)
          self.send_checkpoint(failingtasks, self.args.checkpoint_on_failure)
          self.stop_pipeline_and_exit(process_list)

       # empty finished means we have to wait more
       return len(finished)==0


    def get_logfile(self, tid):
        # determines the logfile name for this task
        taskspec = self.workflowspec['stages'][tid]
        taskname = taskspec['name']
        filename = taskname + '.log'
        directory = taskspec['cwd']
        return directory + '/' + filename


    def is_worth_retrying(self, tid):
        # This checks for some signatures in logfiles that indicate that a retry of this task
        # might have a chance.
        # Ideally, this should be made user configurable. Either the user could inject a lambda
        # or a regular expression to use. For now we just put a hard coded list
        logfile = self.get_logfile(tid)

        return True #! --> for now we just retry tasks a few times

        # 1) ZMQ_EVENT + interrupted system calls (DPL bug during shutdown)
        # Not sure if grep is faster than native Python text search ...
        # status = os.system('grep "failed setting ZMQ_EVENTS" ' + logfile + ' &> /dev/null')
        # if os.WEXITSTATUS(status) == 0:
        #   return True

        # return False


    def cat_logfiles_tostdout(self, taskids):
        # In case of errors we can cat the logfiles for this taskname
        # to stdout. Assuming convention that "taskname" translates to "taskname.log" logfile.
        for tid in taskids:
            logfile = self.get_logfile(tid)
            if os.path.exists(logfile):
                print (' ----> START OF LOGFILE ', logfile, ' -----')
                os.system('cat ' + logfile)
                print (' <---- END OF LOGFILE ', logfile, ' -----')

    def send_checkpoint(self, taskids, location):
        # Makes a tarball containing all files in the base dir
        # (timeframe independent) and the dir with corrupted timeframes
        # and copies it to a specific ALIEN location. Not a core function
        # just some tool get hold on error conditions appearing on the GRID.

        def get_tar_command(dir='./', flags='cf', findtype='f', filename='checkpoint.tar'):
            return 'find ' + str(dir) + ' -maxdepth 1 -type ' + str(findtype) + ' -print0 | xargs -0 tar ' + str(flags) + ' ' + str(filename)

        if location != None:
           print ('Making a failure checkpoint')
           # let's determine a filename from ALIEN_PROC_ID - hostname - and PID

           aliprocid=os.environ.get('ALIEN_PROC_ID')
           if aliprocid == None:
              aliprocid = 0

           fn='pipeline_checkpoint_ALIENPROC' + str(aliprocid) + '_PID' + str(os.getpid()) + '_HOST' + socket.gethostname() + '.tar'
           actionlogger.info("Checkpointing to file " + fn)
           tarcommand = get_tar_command(filename=fn)
           actionlogger.info("Taring " + tarcommand)

           # create a README file with instruction on how to use checkpoint
           readmefile=open('README_CHECKPOINT_PID' + str(os.getpid()) + '.txt','w')

           for tid in taskids:
             taskspec = self.workflowspec['stages'][tid]
             name = taskspec['name']
             readmefile.write('Checkpoint created because of failure in task ' + name + '\n')
             readmefile.write('In order to reproduce with this checkpoint, do the following steps:\n')
             readmefile.write('a) setup the appropriate O2sim environment using alienv\n')
             readmefile.write('b) run: $O2DPG_ROOT/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt ' + name + '$ --retry-on-failure 0\n')
           readmefile.close()

           # first of all the base directory
           os.system(tarcommand)

           # then we add stuff for the specific timeframes ids if any
           for tid in taskids:
             taskspec = self.workflowspec['stages'][tid]
             directory = taskspec['cwd']
             if directory != "./":
               tarcommand = get_tar_command(dir=directory, flags='rf', filename=fn)
               actionlogger.info("Tar command is " + tarcommand)
               os.system(tarcommand)
               # same for soft links
               tarcommand = get_tar_command(dir=directory, flags='rf', findtype='l', filename=fn)
               actionlogger.info("Tar command is " + tarcommand)
               os.system(tarcommand)

           # prepend file:/// to denote local file
           fn = "file://" + fn
           actionlogger.info("Local checkpoint file is " + fn)

           # location needs to be an alien path of the form alien:///foo/bar/
           copycommand='alien.py cp ' + fn + ' ' + str(location) + '@disk:1'
           actionlogger.info("Copying to alien " + copycommand)
           os.system(copycommand)

    def init_alternative_software_environments(self):
        """
        Initiatialises alternative software environments for specific tasks, if there
        is an annotation in the workflow specificiation.
        """

        environment_cache = {}
        # go through all the tasks once and setup environment
        for taskid in range(len(self.workflowspec['stages'])):
          packagestr = self.workflowspec['stages'][taskid].get("alternative_alienv_package")
          if packagestr == None:
             continue

          if environment_cache.get(packagestr) == None:
             environment_cache[packagestr] = get_alienv_software_environment(packagestr)

          self.alternative_envs[taskid] = environment_cache[packagestr]


    def analyse_files_and_connections(self):
        for p,s in self.pid_to_files.items():
            for f in s:
                print("F" + str(f) + " : " + str(p))
        for p,s in self.pid_to_connections.items():
            for c in s:
               print("C" + str(c) + " : " + str(p))
            #print(str(p) + " CONS " + str(c))
        try:
            # check for intersections
            for p1, s1 in self.pid_to_files.items():
                for p2, s2 in self.pid_to_files.items():
                    if p1!=p2:
                        if type(s1) is set and type(s2) is set:
                            if len(s1)>0 and len(s2)>0:
                                try:
                                    inters = s1.intersection(s2)
                                except Exception:
                                    print ('Exception during intersect inner')
                                    pass
                                if (len(inters)>0):
                                    print ('FILE Intersection ' + str(p1) + ' ' + str(p2) + ' ' + str(inters))
          # check for intersections
            for p1, s1 in self.pid_to_connections.items():
                for p2, s2 in self.pid_to_connections.items():
                    if p1!=p2:
                        if type(s1) is set and type(s2) is set:
                            if len(s1)>0 and len(s2)>0:
                                try:
                                    inters = s1.intersection(s2)
                                except Exception:
                                    print ('Exception during intersect inner')
                                    pass
                                if (len(inters)>0):
                                    print ('CON Intersection ' + str(p1) + ' ' + str(p2) + ' ' + str(inters))

            # check for intersections
            #for p1, s1 in slf.pid_to_files.items():
            #    for p2, s2 in self.pid_to_files.items():
            #        if p1!=p2 and len(s1.intersection(s2))!=0:
            #            print ('Intersection found files ' + str(p1) + ' ' + str(p2) + ' ' + s1.intersection(s2))
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('Exception during intersect outer')
            pass

    def is_good_candidate(self, candid, finishedtasks):
        if self.procstatus[candid] != 'ToDo':
            return False
        needs = set([self.tasktoid[t] for t in self.taskneeds[self.idtotask[candid]]])
        if set(finishedtasks).intersection(needs) == needs:
            return True
        return False

    def emit_code_for_task(self, tid, lines):
        actionlogger.debug("Submitting task " + str(self.idtotask[tid]))
        taskspec = self.workflowspec['stages'][tid]
        c = taskspec['cmd']
        workdir = taskspec['cwd']
        env = taskspec.get('env')
        # in general:
        # try to make folder
        lines.append('[ ! -d ' + workdir + ' ] && mkdir ' + workdir + '\n')
        # cd folder
        lines.append('cd ' + workdir + '\n')
        # set local environment
        if env!=None:
            for e in env.items():
                lines.append('export ' + e[0] + '=' + str(e[1]) + '\n')
        # do command
        lines.append(c + '\n')
        # unset local environment
        if env!=None:
            for e in env.items():
                lines.append('unset ' + e[0] + '\n')

        # cd back
        lines.append('cd $OLDPWD\n')


    # produce a bash script that runs workflow standalone
    def produce_script(self, filename):
        # pick one of the correct task orderings
        taskorder = self.topological_orderings[0]
        outF = open(filename, "w")

        lines=[]
        # header
        lines.append('#!/usr/bin/env bash\n')
        lines.append('#THIS FILE IS AUTOGENERATED\n')
        lines.append('export JOBUTILS_SKIPDONE=ON\n')

        # we record the global environment setting
        # in particular to capture global workflow initialization
        lines.append('#-- GLOBAL INIT SECTION FROM WORKFLOW --\n')
        for e in self.globalenv:
            lines.append('export ' + str(e) + '=' + str(self.globalenv[e]) + '\n')
        lines.append('#-- TASKS FROM WORKFLOW --\n')
        for tid in taskorder:
            print ('Doing task ' + self.idtotask[tid])
            self.emit_code_for_task(tid, lines)

        outF.writelines(lines)
        outF.close()


    # print error message when no progress can be made
    def noprogress_errormsg(self):
        # TODO: rather than writing this out here; refer to the documentation discussion this?
        msg = """Scheduler runtime error: The scheduler is not able to make progress although we have a non-zero candidate set.

Explanation: This is typically the case because the **ESTIMATED** resource requirements for some tasks
in the workflow exceed the available number of CPU cores or the memory (as explicitely or implicitely determined from the
--cpu-limit and --mem-limit options). Often, this might be the case on laptops with <=16GB of RAM if one of the tasks
is demanding ~16GB. In this case, one could try to tell the scheduler to use a slightly higher memory limit
with an explicit --mem-limit option (for instance `--mem-limit 20000` to set to 20GB). This might work whenever the
**ACTUAL** resource usage of the tasks is smaller than anticipated (because only small test cases are run).

In addition it might be worthwile running the workflow without this resource aware, dynamic scheduler.
This is possible by converting the json workflow into a linearized shell script and by directly executing the shell script.
Use the `--produce-script myscript.sh` option for this.
"""
        print (msg, file=sys.stderr)

    def execute(self):
        starttime = time.perf_counter()
        psutil.cpu_percent(interval=None)
        os.environ['JOBUTILS_SKIPDONE'] = "ON"
        errorencountered = False

        def speedup_ROOT_Init():
               """initialize some env variables that speed up ROOT init
               and prevent ROOT from spawning many short-lived child
               processes"""

               # only do it on Linux
               if platform.system() != 'Linux':
                  return

               if os.environ.get('ROOT_LDSYSPATH')!=None and os.environ.get('ROOT_CPPSYSINCL')!=None:
                  # do nothing if already defined
                  return

               # a) the PATH for system libraries
               # search taken from ROOT TUnixSystem
               cmd='LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls /tmp/DOESNOTEXIST 2>&1 | grep -m 1 "system search path" | sed \'s/.*=//g\' | awk \'//{print $1}\''
               proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
               libpath, err = proc.communicate()
               if not (args.no_rootinit_speedup == True):
                  print ("setting up ROOT system")
                  os.environ['ROOT_LDSYSPATH'] = libpath.decode()

               # b) the PATH for compiler includes needed by Cling
               cmd='LC_ALL=C c++ -xc++ -E -v /dev/null 2>&1 | sed -n \'/^.include/,${/^ \/.*++/{p}}\'' # | sed \'s/ //\''
               proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
               incpath, err = proc.communicate()
               incpaths = [ line.lstrip() for line in incpath.decode().splitlines() ]
               joined = ':'.join(incpaths)
               if not (args.no_rootinit_speedup == True):
                  os.environ['ROOT_CPPSYSINCL'] = joined

        speedup_ROOT_Init()

        # we make our own "tmp" folder
        # where we can put stuff such as tmp socket files etc (for instance DPL FAIR-MQ sockets)
        # (In case of running within docker/singularity, this may not be so important)
        if not os.path.isdir("./.tmp"):
          os.mkdir("./.tmp")
        if os.environ.get('FAIRMQ_IPC_PREFIX')==None:
          socketpath = os.getcwd() + "/.tmp"
          actionlogger.info("Setting FAIRMQ socket path to " + socketpath)
          os.environ['FAIRMQ_IPC_PREFIX'] = socketpath

        # some maintenance / init work
        if args.list_tasks:
          print ('List of tasks in this workflow:')
          for i,t in enumerate(self.workflowspec['stages'],0):
              print (t['name'] + '  (' + str(t['labels']) + ')' + ' ToDo: ' + str(not self.ok_to_skip(i)))
          exit (0)

        if args.produce_script != None:
          self.produce_script(args.produce_script)
          exit (0)

        if args.rerun_from:
          reruntaskfound=False
          for task in self.workflowspec['stages']:
              taskname=task['name']
              if re.match(args.rerun_from, taskname):
                reruntaskfound=True
                taskid=self.tasktoid[taskname]
                self.remove_done_flag(find_all_dependent_tasks(self.possiblenexttask, taskid))
          if not reruntaskfound:
              print('No task matching ' + args.rerun_from + ' found; cowardly refusing to do anything ')
              exit (1)

        # *****************
        # main control loop
        # *****************
        candidates = [ tid for tid in self.possiblenexttask[-1] ]

        self.process_list=[] # list of tuples of nodes ids and Popen subprocess instances

        finishedtasks=[] # global list of finished tasks

        try:

            while True:
                # sort candidate list according to task weights
                candidates_tmp = candidates.copy()
                candidates = []
                for tid in candidates_tmp:
                    self.update_resources(tid)
                    candidates.append((tid, self.taskweights[tid], self.cpuperid[tid] / self.cpulimit))

                if args.optimise_cpu:
                    candidates.sort(key=lambda tup: (-tup[2], -tup[1][1])) # prefer heavy tasks, after that sort by importance
                else:
                    candidates.sort(key=lambda tup: (tup[1][0],-tup[1][1])) # prefer small and same timeframes first then prefer important tasks within frameframe
                # remove weights
                candidates = [ tid for tid,_,_ in candidates ]

                finished = [] # --> to account for finished because already done or skipped
                actionlogger.debug('Sorted current candidates: ' + str([(c,self.idtotask[c]) for c in candidates]))
                self.try_job_from_candidates_wrapper(candidates, finished)
                if len(candidates) > 0 and len(self.process_list) == 0:
                    if self.curcpubooked != 0 or self.curcpubooked_backfill != 0:
                        self.curcpubooked = 0
                        self.curcpubooked_backfill = 0
                        continue

                    self.noprogress_errormsg()
                    send_webhook(self.args.webhook,"Unable to make further progress: Quitting")
                    errorencountered = True
                    break

                finished_from_started = [] # to account for finished when actually started
                failing = []
                while self.waitforany(self.process_list, finished_from_started, failing):
                    if not args.dry_run:
                        self.monitor(self.process_list) #  ---> make async to normal operation?
                        time.sleep(1) # <--- make this incremental (small wait at beginning)
                    else:
                        time.sleep(0.001)

                finished = finished + finished_from_started
                actionlogger.debug("finished now :" + str(finished_from_started))
                finishedtasks = finishedtasks + finished

                # if a task was marked "failed" and we come here (because
                # we use --keep-going) ... we need to take out the pid from finished
                if len(failing) > 0:
                    # remove these from those marked finished in order
                    # not to continue with their children
                    errorencountered = True
                    for t in failing:
                        finished = [ x for x in finished if x != t ]
                        finishedtasks = [ x for x in finishedtasks if x != t ]

                # if a task was marked as "retry" we simply put it back into the candidate list
                if len(self.tids_marked_toretry) > 0:
                    # we need to remove these first of all from those marked finished
                    for t in self.tids_marked_toretry:
                        finished = [ x for x in finished if x != t ]
                        finishedtasks = [ x for x in finishedtasks if x != t ]

                    candidates = candidates + self.tids_marked_toretry
                    self.tids_marked_toretry = []


                # new candidates
                for tid in finished:
                    if self.possiblenexttask.get(tid)!=None:
                        potential_candidates=list(self.possiblenexttask[tid])
                        for candid in potential_candidates:
                        # try to see if this is really a candidate:
                            if self.is_good_candidate(candid, finishedtasks) and candidates.count(candid)==0:
                                candidates.append(candid)

                actionlogger.debug("New candidates " + str( candidates))
                send_webhook(self.args.webhook, "New candidates " + str(candidates))

                if len(candidates)==0 and len(self.process_list)==0:
                   break
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            traceback.print_exc()
            print ('Cleaning up ')

            self.SIGHandler(0,0)

        endtime = time.perf_counter()
        statusmsg = "success"
        if errorencountered:
           statusmsg = "with failures"

        print ('\n**** Pipeline done ' + statusmsg + ' (global_runtime : {:.3f}s) *****\n'.format(endtime-starttime))
        actionlogger.debug("global_runtime : {:.3f}s".format(endtime-starttime))
        return errorencountered


if args.cgroup!=None:
    myPID=os.getpid()
    # cgroups such as /sys/fs/cgroup/cpuset/<cgroup-name>/tasks
    # or              /sys/fs/cgroup/cpu/<cgroup-name>/tasks
    command="echo " + str(myPID) + f" > {args.cgroup}"
    actionlogger.info("applying cgroups " + command)
    os.system(command)

executor=WorkflowExecutor(args.workflowfile,jmax=args.maxjobs,args=args)
exit (executor.execute())
