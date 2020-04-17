from .showAttendTell import ShowAttendTell
from .utils import *

def get_model(name, **kwargs):
    if name == 'ShowAttendTell':
        return ShowAttendTell(**kwargs)