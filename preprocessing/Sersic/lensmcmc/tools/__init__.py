import yaml
import numpy as np
import os
from importlib import import_module
from argparse import ArgumentParser


class RunParser(ArgumentParser):

    def __init__(self):

        # Set up cmd line parsing
        super().__init__()
        self.add_argument("dir", help="Directory for the run")
        self.add_argument("-o", "--observation", type=str,
                          help="Observation settings file")
        self.add_argument("-m", "--model", type=str,
                          help="Model settings file")
        self.add_argument("-p", "--processing", action="store_false",
                          help="Disable MPI")
        self.add_argument("-r", "--resume", action="store_true",
                          help="Continue a previous run")

    def parse_args_(self):
        args = self.parse_args()
        target = args.dir
        args.dir = os.environ['RESULTS'] + '/' + target
        return args


def mkdir_check(path):
    """
    Makes a directory if it does not already exist
    """
    if not os.path.exists(path):
        os.mkdir(path)


def check_master(mpipool):
    if not mpipool:
        return True
    else:
        if mpipool.is_master():
            return True
        else:
            return False


def generic_yaml_loader(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def observation_yaml_loader(file):
    """
    Loads the parameters and model objects
    from the yaml file
    """
    if type(file) is dict:
        p = file
    else:
        with open(file, 'r') as f:
            p = yaml.load(f, Loader=yaml.FullLoader)

    # Add lensing strength
    if 'lensing_strength' not in p['lens'].keys():
        p['lens']['lensing_strength'] = (p['lens']['einstein_radius'] *
                                         np.sqrt(p['lens']['axis_ratio']))

    # Add break radius
    if 'nu_ratio' in p['lens'].keys():
        p['lens']['break_radius'] = p['lens']['lensing_strength'] * p['lens']['nu_ratio']

    # Load mass model
    lens_module = import_module('lensmcmc.models.masses.' + p['lens']['lens_model'])
    mass = getattr(lens_module, p['lens']['lens_model'])

    # Load source model
    src_module = import_module('lensmcmc.models.sources.' + p['src']['source_model'])
    source = getattr(src_module, p['src']['source_model'])

    return p, mass, source


def fitting_yaml_loader(file, obs=None):

    # Load model settings
    with open(file, 'r') as f:
        p = yaml.load(f, Loader=yaml.FullLoader)

    # Add params if needed
    if not p['model']['inherit']:

        # Add lensing strength
        if 'lensing_strength' not in p['model']['params']['lens'].keys():
            p['model']['params']['lens']['lensing_strength'] = (
                    p['model']['params']['lens']['einstein_radius'] *
                    np.sqrt(p['model']['params']['lens']['axis_ratio'])
            )

        # Add break radius
        if 'nu_ratio' in p['model']['params']['lens'].keys():
            p['model']['params']['lens']['break_radius'] = (
                    p['model']['params']['lens']['lensing_strength'] *
                    p['model']['params']['lens']['nu_ratio']
            )

        # Add ellipticity/position angle transforms to lens and source
        for key in ['lens', 'src']:
            ellp = 1.0 - p['model']['params'][key]['axis_ratio']
            posa = p['model']['params'][key]['position_angle']
            p['model']['params'][key]['ellipticity_x'] = ellp * np.cos(posa)
            p['model']['params'][key]['ellipticity_y'] = ellp * np.sin(posa)

    # Add ellipticity/position angle transforms to lens and source
    # and copy observation parameters to fitting parameters
    else:

        for key in ['lens', 'src']:
            for key_ in obs.p[key].keys():
                if type(obs.p[key][key_]) != str:
                    p['model']['params'][key][key_] = obs.p[key][key_]

            ellp = 1.0 - obs.p[key]['axis_ratio']
            posa = obs.p[key]['position_angle']
            p['model']['params'][key]['ellipticity_x'] = ellp * np.cos(posa)
            p['model']['params'][key]['ellipticity_y'] = ellp * np.sin(posa)

    return p
