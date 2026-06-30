import multiprocessing
import os

# Bind to 0.0.0.0 on the port specified by the PORT environment variable
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Neural network inference is CPU-heavy. To prevent OOM and CPU thrashing, 
# keep worker count low and use threads.
workers = int(os.environ.get('WEB_CONCURRENCY', 2))
threads = 4
timeout = 120

# Preload application to load PyTorch weights in the master process 
# before workers are forked. This utilizes Copy-on-Write memory sharing 
# and dramatically reduces RAM utilization per worker.
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
