#!/usr/bin/env python3


import importlib.util

spec = importlib.util.spec_from_file_location("o2dpg_sim_workflow", join(dirname(__file__), "o2dpg_sim_workflow.py"))
o2dpg_sim_workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o2dpg_sim_workflow)

# for dummy timestamp
import time
import logging


class ExitHandler(logging.Handler):
    """
    Add custom logging handler to exit on certain logging level
    """
    def emit(self, record):
        logging.shutdown()
        sys.exit(1)

class O2DPGLoggerFormatter(logging.Formatter):
    """
    A custom formatter that colors the levelname on request
    """
    # color names to indices
    color_map = {
        'black': 0,
        'red': 1,
        'green': 2,
        'yellow': 3,
        'blue': 4,
        'magenta': 5,
        'cyan': 6,
        'white': 7,
    }

    level_map = {
        logging.DEBUG: (None, 'blue', False),
        logging.INFO: (None, 'black', False),
        logging.WARNING: (None, 'yellow', False),
        logging.ERROR: (None, 'red', False),
        logging.CRITICAL: ('red', 'white', True),
    }
    csi = '\x1b['
    reset = '\x1b[0m'

    # Define default format string
    def __init__(self, fmt='%(levelname)s in %(pathname)s:%(lineno)d:\n%(message)s',
                 datefmt=None, style='%', color=False):
        logging.Formatter.__init__(self, fmt, datefmt, style)
        self.color = color

    def format(self, record):
        # Copy the record so the global format is kept
        cached_record = copy(record)
        requ_color = self.color
        # Could be a lambda so check for callable property
        if callable(self.color):
            requ_color = self.color()
        # Make sure levelname takes same space for all cases
        cached_record.levelname = f"{cached_record.levelname:8}"
        # Colorize if requested
        if record.levelno in self.level_map and requ_color:
            bg, fg, bold = self.level_map[record.levelno]
            params = []
            if bg in self.color_map:
                params.append(str(self.color_map[bg] + 40))
            if fg in self.color_map:
                params.append(str(self.color_map[fg] + 30))
            if bold:
                params.append('1')
            if params:
                cached_record.levelname = "".join((self.csi, ';'.join(params), "m",
                                                   cached_record.levelname,
                                                   self.reset))
        return logging.Formatter.format(self, cached_record)



def configure_logger(name, level=logging.INFO, logfile=None):
    """
    Basic configuration adding a custom formatted StreamHandler and turning on
    debug info if requested.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    # Turn on debug info only on request
    logger.setLevel(level)

    sh = logging.StreamHandler()
    formatter = O2DPGLoggerFormatter(color=lambda : getattr(sh.stream, 'isatty', None))

    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Add logfile on request
    if logfile is not None:
        # Specify output format
        fh = logging.FileHandler(logfile)
        fh.setFormatter(O2DPGLoggerFormatter())
        logger.addHandler(fh)

    # Add handler to exit at critical. Do this as the last step so all former
    # logger flush before aborting
    logger.addHandler(ExitHandler(logging.CRITICAL))

    return logger


def get_logger(name="O2DPG-MC-Production"):
    """
    Get the global logger for this package and set handler together with formatters.
    """
    rturn configure_logger(name)

def sample_timestamps(grp_objects, timestamp_sampling, prod_fraction):
    return []



def collect_grp_objects(ccdb_url, *, run_numbers=None, periods=None, years=None):
    if not run_numbers and not periods and not years:
        return None
    # 1. collect all years
    # 2. union with periods
    # 3. union with single run numbers

    # Return a list of GRP objects
    return []


def make_unanchored_workflow(args):
    # This will simply dump a workflow.json
    return o2dpg_sim_workflow.make_workflow(args)



def sample_timestamps(grp, prod_fraction, timestamp_sampling):
    # this for now just returns a number of dummy timestamps
    return 0, 0, [int(time.time() * 1000) for _ in range(timestamp_sampling)]


def make_anchored_workflow(grp, args):

    # sample timestamps
    # tag is passed so we can check if another production with the same tag was done already. In that case we can derive another sub-tag
    run_number, cycle, timestamps = sample_timestamps(grp, args.tag, args.prod_fraction, args.timestamp_sampling)
    # some kind of sub tag For instance:
    # assume was run for this tag with certain fraction. Now run again with some fraction --> we want to fill some gaps
    # this will now get the next sub tag
    # sub tag can only be derived once the run number is known
    for i, ts in enumerate(timestamps):
        args.timestamp = ts
        args.o = f"mc_prod_RN_{run_number}_TAG_{args.tag}_CYCLE_{cycle}_ID_{i}_TS_{ts}
        return o2dpg_sim_workflow.make_workflow(args)


def run(args):

    logger = get_logger()

    check_envs_list = ["O2DPG_ROOT", "O2_ROOT", "O2PHYSICS_ROOT"]
    if args.include_qc:
        check_envs_list.append("QUALITYCONTROL_ROOT")
    for cel in check_envs_list:
        if not environ.get(cel, None):
            logger.error("Need all environments loaded:")
            for e in check_envs_list:
                print(f"  {e}")
            return 1

    if not args.run_number and not args.period and not args.year:
        logger.info("Un-anchored run requested")
        return make_unanchored_workflow(args)

    # READY in case that was un-anchored

    # GO ON if anchored
    if not args.tag:
        logger.error("Tag required for anchored run")
        return 1

    grp_objects = collect_grp_objects(args.ccdb_url, run_numbers=args.run_number, periods=args.period, years=args.year)

    if not grp_objects:
        logger.error("Cannot retrieve GRPs for requested anchored simulation")
        return 1

    for grp in grp_objects:
        make_anchored_workflow(gpr, args)

    return 0


def main():

    # Eventually we take the same argument parser used by the sim workflow
    sim_wf_parser = o2dpg_sim_workflow.PARSER

    sim_wf_parser.add_argument("--run-number", dest="run_number", type=int, nargs="*", help="run number to anchor production to (no anchoring if nothing given)")
    sim_wf_parser.add_argument("--period", type=str, nargs="*", help="anchor to a period (assumed to be a set of run numbers)")
    sim_wf_parser.add_argument("--year", type=str, nargs="*", help="anchor to a year (assumed to be a set of periods)")
    sim_wf_parser.add_argument("--prod-fraction", dest="prod_fraction", type=float, default=0.1, help="fraction of luminosity the simulation should corespond to")
    sim_wf_parser.add_argument("--timestamp-sampling", dest="timestamp_sampling" type=int, default=10, help="how many timestamps to be sampled")
    sim_wf_parser.add_argument("--ccdb-url", dest="ccdb_url", type=str, default="ccdb-test.cern.ch:8080", help="URL to CCDB to be queried")
    sim_wf_parser.add_argument("--tag", help="a tag, required if anchored run")

    args = sim_wf_parser.parse_args()

    return run(args)




if __name__ == "__main__":
    sys.exit(main())
