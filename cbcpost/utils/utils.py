import keyword
from time import time

def on_master_process():
    return MPI.process_number() == 0

def in_serial():
    return MPI.num_processes() == 1


def strip_code(code):
    """Strips code of unnecessary spaces, comments etc."""
    s = []

    code = code.split('\n')
    for i, l in enumerate(reversed(code)):
        if l.count(')') > l.count('('):
            code[-2-i] += " "+l.strip(' ')
            code[-1-i] = ''

    for l in code:
        l = l.split('#')[0]
        l = l.replace('\t', '    ')
        l = l.replace('    ', ' ')

        l = l.rstrip(' ')        
        l = l.split(' ')
        
        l_new = ""
        string_flag = False
        for c in l:
            if c.count('"') == 1 or c.count("'") == 1:
                if string_flag:
                    l_new += " "+c
                else:
                    l_new += c
                string_flag = not string_flag
            elif string_flag:
                l_new += " "+c
            elif c == '':
                l_new += ' '
            elif c in keyword.kwlist:
                if l_new[-1] == " ":
                    l_new += c+" "
                else:
                    l_new += " "+c+" "
            else:
                l_new += c

        if l_new.strip(' ') != '':
            s.append(l_new)

    s = '\n'.join(s)
    return s

# --- I/O stuff ---
class _HDF5Link:
    cpp_link_module = None
    def __init__(self):
        cpp_link_code = '''
        #include <hdf5.h>
        void link_dataset(const std::string hdf5_filename,
                                  const std::string link_from,
                                  const std::string link_to)
        {
            hid_t hdf5_file_id = HDF5Interface::open_file(hdf5_filename, "a", true);
        
            herr_t status = H5Lcreate_hard(hdf5_file_id, link_from.c_str(), H5L_SAME_LOC,
                                link_to.c_str(), H5P_DEFAULT, H5P_DEFAULT);
            dolfin_assert(status != HDF5_FAIL);
            
            HDF5Interface::close_file(hdf5_file_id);        
        }
        '''
        
        self.cpp_link_module = compile_extension_module(cpp_link_code, additional_system_headers=["dolfin/io/HDF5Interface.h"])
    
    def link(self, hdf5filename, link_from, link_to):
        self.cpp_link_module.link_dataset(hdf5filename, link_from, link_to)
hdf5_link = _HDF5Link().link


def safe_mkdir(dir):
    """Create directory without exceptions in parallel."""
    # Create directory
    if not os.path.isdir(dir):
        try:
            os.makedirs(dir)
        except:
            # Allow race condition when multiple processes
            # work in same directory, ignore exception.
            pass

    # Wait for all processes to finish, hopefully somebody
    # managed to create the directory...
    MPI.barrier()

    # Warn if this failed
    if not os.path.isdir(dir):
        #warning("FAILED TO CREATE DIRECTORY %s" % (dir,))
        Exception("FAILED TO CREATE DIRECTORY %s" % (dir,))

# --- Logging ---

def cbc_warning(msg):
    if on_master_process():
        warning(msg)

def cbc_print(msg):
    if on_master_process():
        print msg

def cbc_log(level, msg):
    if on_master_process():
        log(level, msg)


# --- System inspection ---

from os import getpid
from commands import getoutput
def get_memory_usage():
    """Return memory usage in MB"""
    try:
        from fenicstools import getMemoryUsage
        return getMemoryUsage()
    except:
        cbc_warning("Unable to load fenicstools to check memory usage. Falling back to unsafe memory check.")
        mypid = getpid()
        mymemory = getoutput("ps -o rss %s" % mypid).split()[1]
        return int(mymemory)/1024

# --- Timing ---

class Timer:
    def __init__(self, frequency=0):
        self._frequency = frequency
        self._timer = time()
        self._timings = {}
        self._keys = []
        self._N = 0

    def completed(self, key, summables={}):
        if self._frequency == 0:
            return
        
        if key not in self._timings:
            self._keys.append(key)
            self._timings[key] = [0,0, {}]
        
        t = time()
        ms = (t - self._timer)*1000
        self._timings[key][0] += ms
        self._timings[key][1] += 1
        
        for k,v in summables.items():
            if k not in self._timings[key][2]:
                self._timings[key][2][k] = 0
            self._timings[key][2][k] += v
        
        if self._frequency == 1:
            s = "%10.0f ms: %s" % (ms, key)
            ss = []
            #if summables != {}:
            #    s += "  ("
            for k, v in summables.items():
                #s += "%s: %s, " %(k,v)
                ss.append("%s=%s" %(k,v))
            if len(ss) > 0:
                ss = "  ("+", ".join(ss)+")"
            else:
                ss = ""
            s += ss

            cbc_print(s)

        self._timer = time()
    
    def _print_summary(self):
        cbc_print("Timings summary: ")
        
        for key in self._keys:
            tot = self._timings[key][0]
            N = self._timings[key][1]
            avg = int(1.0*tot/N)
            
            s = "%10.0f ms (avg: %8.0f ms, N: %5d): %s" %(tot, avg, N, key)
            
            summables = self._timings[key][2]
            ss = []
            #if summables != {}:
            #    s += "("
            for k, tot in summables.items():
                avg = int(1.0*tot/N)
                #s += "%s: %s (avg: %s), " %(k,tot,avg)
                ss.append("%s=%s (avg: %s)" %(k,tot,avg))
            #if summables != {}:
            #    s += ")"
            if len(ss) > 0:
                ss = "  ("+", ".join(ss)+")"
            else:
                ss = ""
            s += ss
            cbc_print(s)
    
    def _reset(self):
        self._timings = {}
        self._keys = []
        self._N = 0
        
    def increment(self):
        self._N += 1
        if self._frequency > 1 and self._N % self._frequency == 0:
            self._print_summary()
            self._reset()

def timeit(t0=None, msg=None):
    if t0 is None:
        return time()
    else:
        t = time() - t0
        cbc_print("%s: %g" % (msg, t))
        return t


