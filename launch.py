from workspace import *
from render import *
from threading import Thread


box = WorkSpace()

# thread = Thread(target=Render(box))
# thread.daemon = True
# thread.start()

Render(box)

