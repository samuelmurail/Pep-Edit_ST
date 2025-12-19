import argparse
import numpy as np
import time
import os
import sys
import logging
import pdb_numpy
import openmm
from openmm import LangevinMiddleIntegrator, unit, Platform, app

from openff import nagl_models
from openff.nagl import GNNModel

from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openff.toolkit import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import SST2
from SST2.st import run_st
import SST2.tools as tools


# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add sys.sdout as handler
logger.addHandler(logging.StreamHandler(sys.stdout))

def parser_input():

    # Parse arguments :
    parser = argparse.ArgumentParser(
        description='Simulate a peptide starting from a linear conformation.')
    parser.add_argument('-SMILES', action="store", dest="smiles",
                        help='Input SMILES file', type=str, required=True)
    parser.add_argument('-pdb', action="store", dest="pdb",
                        help='Input PDB file', type=str, required=True)
    parser.add_argument('-n', action="store", dest="name",
                        help='Output file name', type=str, required=True)
    parser.add_argument('-dir', action="store", dest="out_dir",
                        help='Output directory for intermediate files',
                        type=str, required=True)
    parser.add_argument('-pad', action="store", dest="pad",
                        help='Box padding, default=1.5 nm',
                        type=float,
                        default=1.5)
    parser.add_argument('-eq_time_expl',
                        action="store",
                        dest="eq_time_expl",
                        help='Explicit Solvent Equilibration time, default=10 (ns)',
                        type=float,
                        default=10)
    parser.add_argument('-time',
                        action="store",
                        dest="time",
                        help='ST time, default=10.000 (ns)',
                        type=float,
                        default=10000)
    parser.add_argument('-temp_list',
                        action="store",
                        dest="temp_list",
                        nargs='+',
                        help='SST2 temperature list, default=None',
                        type=float,
                        default=None)
    parser.add_argument('-temp_time',
                        action="store",
                        dest="temp_time",
                        help='ST temperature time change interval, default=2.0 (ps)',
                        type=float,
                        default=2.0)
    parser.add_argument('-log_time',
                        action="store",
                        dest="log_time",
                        help='ST log save time interval, default= temp_time=2.0 (ps)',
                        type=float,
                        default=None)
    parser.add_argument('-min_temp',
                        action="store",
                        dest="min_temp",
                        help='Base temperature, default=300(K)',
                        type=float,
                        default=300)
    parser.add_argument('-last_temp',
                        action="store",
                        dest="last_temp",
                        help='Base temperature, default=500(K)',
                        type=float,
                        default=500)
    parser.add_argument('-hmr',
                        action="store",
                        dest="hmr",
                        help='Hydrogen mass repartition, default=3.0 a.m.u.',
                        type=float,
                        default=3.0)
    parser.add_argument('-temp_num',
                        action="store",
                        dest="temp_num",
                        help='Temperature rung number, default=None (computed as function of Epot)',
                        type=int,
                        default=None)
    parser.add_argument('-friction',
                        action="store",
                        dest="friction",
                        help='Langevin Integrator friction coefficient default=10.0 (ps-1)',
                        type=float,
                        default=10.0)
    parser.add_argument('-ff',
                        action="store",
                        dest="ff",
                        help='force field, default=amber14',
                        default='amber14sb')
    parser.add_argument('-water_ff',
                        action="store",
                        dest="water_ff",
                        help='force field, default=tip3p',
                        default='tip3p')
    parser.add_argument('-v',
                        action='store_true',
                        dest="verbose",
                        help='Verbose mode')


    return parser


def prepare_smiles_pdb(smiles_string: str, pdb_file: str,out_pdb: str) -> Molecule:
    
    # Parse SMILES
    logger.info(f"- Generate 3D conformation from SMILES")
    start_time = time.time()
    with open(smiles_string, 'r') as f:
        smiles = f.read().strip()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    # params = AllChem.ETKDGv3()
    # params.randomSeed = 0xF00D  # set for reproducibility; remove for stochastic behavior
    # params.maxAttempts = 10000
    AllChem.EmbedMolecule(
        mol,
        maxAttempts=10000,
        randomSeed=0xF00D,
        ETversion=2
        )
    
    end_time = time.time()
    logger.info(f"- SMILES parsed in {end_time - start_time:.3f} seconds")


    # Optimize geometry
    logger.info(f"- Optimize SMILES 3D conformation")
    start_time = time.time()

    if AllChem.MMFFHasAllMoleculeParams(mol):
        opt_ret = AllChem.MMFFOptimizeMolecule(mol)
    else:
        opt_ret = AllChem.UFFOptimizeMolecule(mol)
    if opt_ret < 0:
        raise RuntimeError("Geometry optimization failed")

    end_time = time.time()
    logger.info(f"- Geometry optimized in {end_time - start_time:.3f} seconds")

    # Parse PDB
    coor = pdb_numpy.Coor(pdb_file)

    coor_string = pdb_numpy.format.pdb.get_pdb_string(coor)
    pdb_lig = Chem.MolFromPDBBlock(coor_string)
    pdb_lig = AllChem.AssignBondOrdersFromTemplate(mol, pdb_lig)

    # Set RDKit molecule coordinates to those from PDB
    conf = mol.GetConformer()
    conf_pdb = pdb_lig.GetConformer()

    logger.info(f"- Align SMILES conformation to PDB conformation")
    start_time = time.time()

    mapping = mol.GetSubstructMatch(pdb_lig)

    for i, j in enumerate(mapping):
        conf.SetAtomPosition(j, conf_pdb.GetAtomPosition(i))

    end_time = time.time()
    logger.info(f"- Alignment done in {end_time - start_time:.3f} seconds")

    # Add hydrogens (needed for good 3D geometry)
    mol_h = Chem.AddHs(mol, addCoords=True)

    # if AllChem.MMFFHasAllMoleculeParams(mol_h):
    #     opt_ret = AllChem.MMFFOptimizeMolecule(mol_h)
    # else:
    #     opt_ret = AllChem.UFFOptimizeMolecule(mol_h)
    # if opt_ret < 0:
    #     raise RuntimeError("Geometry optimization failed")

    # Write PDB and return an OpenFF Molecule
    Chem.MolToPDBFile(mol_h, out_pdb)
    off_mol = Molecule.from_rdkit(mol_h, allow_undefined_stereo=True)

    logger.info(f"- Compute partial charges with NAGL model")
    start_time = time.time()

    path_models = nagl_models.list_available_nagl_models()
    print(path_models)
    model_rc3 = path_models[-1] # openff-gnn-am1bcc-1.0.0.pt
    model = GNNModel.load(model_rc3)

    charges = model.compute_property(off_mol)
    print(f'Total charge: {sum(charges)}')
    off_mol.partial_charges = unit.Quantity(charges, unit.elementary_charge)

    end_time = time.time()
    logger.info(f"- Partial charges computed in {end_time - start_time:.3f} seconds")

    return off_mol

def create_water_box(
    in_file,
    out_file,
    forcefield,
    pad=None,
    vec=None,
    ionicStrength=0.15 * unit.molar,
    positiveIon="Na+",
    negativeIon="Cl-",
    overwrite=False,
):
    """Add a water box around a prepared cif file.

    Parameters
    ----------
    in_file : str
        Path to the input cif/pdb file
    out_file : str
        Path to the output cif/pdb file
    forcefield : openmm ForceField
        forcefield object
    pad : float
        Padding around the peptide in nm
    vec : float
        Vector of the box (nm), default is None
    ionicStrength : unit.Quantity
        Ionic strength of the system, default is 0.15 M
    positiveIon : str
        Positive ion, default is Na+
    negativeIon : str
        Negative ion, default is Cl-
    overwrite : bool
        Overwrite the output file, default is False
    """

    if vec is None and pad is None:
        raise ValueError("Either pad or vec must be defined")
    if vec is not None and pad is not None:
        raise ValueError("Either pad or vec must be defined")

    if unit.is_quantity(pad):
        pad = pad.in_units_of(unit.nanometer)
    else:
        if pad is not None:
            pad = pad * unit.nanometer

    if unit.is_quantity(vec):
        vec = vec.in_units_of(unit.nanometer)
    else:
        if vec is not None:
            vec = vec * unit.nanometer

    if in_file.lower().endswith(".pdb"):
        cif = app.PDBFile(in_file)
    elif in_file.lower().endswith(".cif") or in_file.lower().endswith(".mmcif"):
        cif = app.PDBxFile(in_file)
    else:
        raise ValueError("Input file must be a pdb or cif file")

    if not overwrite and os.path.isfile(out_file):
        logger.info(f"File {out_file} exists already, skip create_water_box() step")
        if out_file.lower().endswith(".pdb"):
            cif = app.PDBFile(out_file)
        else:
            cif = app.PDBxFile(out_file)
        return cif

    # To avoid issue with clash with residues out of the box:
    x_min = min([0 * unit.nanometer] + [pos[0] for pos in cif.positions])
    y_min = min([0 * unit.nanometer] + [pos[1] for pos in cif.positions])
    z_min = min([0 * unit.nanometer] + [pos[2] for pos in cif.positions])
    min_vec = (
        openmm.Vec3(
            x_min.value_in_unit(unit.nanometer),
            y_min.value_in_unit(unit.nanometer),
            z_min.value_in_unit(unit.nanometer),
        )
        * unit.nanometer
    )
    cif.positions = [
        (pos - min_vec).value_in_unit(unit.nanometer) for pos in cif.positions
    ] * unit.nanometer

    logger.info('Start app.Modeller()')
    start_time = time.time()

    modeller = app.Modeller(cif.topology, cif.positions)

    end_time = time.time()
    logger.info(f"- Modeller created in {end_time - start_time:.3f} seconds")

    # Create Box

    maxSize = max(
        max((pos[i] for pos in cif.positions)) - min((pos[i] for pos in cif.positions))
        for i in range(3)
    )
    vectors = [
        openmm.Vec3(1, 0, 0),
        openmm.Vec3(1 / 3, 2 * unit.sqrt(2) / 3, 0),
        openmm.Vec3(-1 / 3, unit.sqrt(2) / 3, unit.sqrt(6) / 3),
    ]

    if vec is None:
        boxVectors = [(maxSize + pad) * v for v in vectors]
    else:
        boxVectors = [vec * v for v in vectors]
    logger.info(
        f"- Adding solvent with a {boxVectors[0][0].value_in_unit(unit.nanometer):.3} nm size box"
    )

    logger.info('Start modeller.addSolvent')
    start_time = time.time()

    modeller.addSolvent(
        forcefield,
        boxVectors=boxVectors,
        ionicStrength=ionicStrength,
        positiveIon=positiveIon,
        negativeIon=negativeIon,
    )

    end_time = time.time()
    logger.info(f"- Solvent added in {end_time - start_time:.3f} seconds")

    # Save
    if out_file.lower().endswith(".pdb"):
        app.PDBFile.writeFile(
            modeller.topology, modeller.positions, open(out_file, "w"), True
        )
        cif = app.PDBFile(out_file)

    elif out_file.lower().endswith(".cif") or out_file.lower().endswith(".mmcif"):
        app.PDBxFile.writeFile(
            modeller.topology, modeller.positions, open(out_file, "w"), True
        )
        cif = app.PDBxFile(out_file)

    else:
        raise ValueError("Output file must be a pdb or cif file")
    
    return cif


if __name__ == "__main__":

    my_parser = parser_input()
    args = my_parser.parse_args()
    logger.info(args)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode activated")
        SST2.show_log()

    OUT_PATH = args.out_dir
    name = args.name

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    logger.info(f"- Prepare ligand from SMILES using rdkit and openff")
    ligand_molecule = prepare_smiles_pdb(args.smiles, args.pdb, f"{OUT_PATH}/{name}_fixed.pdb")

    forcefield = tools.get_forcefield(args.ff, args.water_ff)

    smir = SMIRNOFFTemplateGenerator(molecules=ligand_molecule)
    forcefield.registerTemplateGenerator(smir.generator)

    logger.info(f"- Create water box")

    # Don't use cif format here for output

    cif = create_water_box(
        f"{OUT_PATH}/{name}_fixed.pdb",
        f"{OUT_PATH}/{name}_water.pdb",
        pad=args.pad,
        forcefield=forcefield,
        overwrite=False)

    ###########################
    ### BASIC EQUILIBRATION ###
    ###########################

    dt = 4 * unit.femtosecond
    temperature = args.min_temp * unit.kelvin
    max_temp = args.last_temp * unit.kelvin
    friction = args.friction / unit.picoseconds
    hydrogenMass = args.hmr * unit.amu
    rigidWater = True
    ewaldErrorTolerance = 0.0005
    nsteps = int(np.ceil(args.eq_time_expl * unit.nanoseconds / dt))

    # pdb = PDBFile(f"{OUT_PATH}/{name}_water.pdb")
    # PDBxFile.writeFile(
    #     pdb.topology,
    #     pdb.positions,
    #     open(f"{OUT_PATH}/{name}_water.cif", "w"),
    #     True)
    # cif = PDBxFile(f"{OUT_PATH}/{name}_water.cif")
    
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)

    logger.info(f"- Create system")

    system = tools.create_sim_system(cif,
        forcefield=forcefield,
        temp=temperature,
        h_mass=args.hmr,
        base_force_group=1)
    
    # Simulation Options
    platform = Platform.getPlatformByName('CUDA')
    #platform = Platform.getPlatformByName('OpenCL')
    platformProperties = {'Precision': 'single'}

    simulation = app.Simulation(
        cif.topology, system, 
        integrator, 
        platform, 
        platformProperties)
    simulation.context.setPositions(cif.positions)

    logger.info(f"- Minimize system")
    
    tools.minimize(
        simulation,
        f"{OUT_PATH}/{name}_em_water.cif",
        cif.topology,
        maxIterations=10000,
        overwrite=False)
    
    simulation.context.setVelocitiesToTemperature(temperature)

    save_step_log = 10000
    save_step_dcd = 10000
    tot_steps = int(np.ceil(args.eq_time_expl * unit.nanoseconds / dt))

    logger.info(f"- Launch equilibration")
    tools.simulate(
        simulation,
        cif.topology,
        tot_steps=tot_steps,
        dt=dt,
        generic_name=f"{OUT_PATH}/{name}_explicit_equi",
        save_step_log = save_step_log,
        save_step_dcd = save_step_dcd,
        )

    ##################
    # ##  ST SIM  ####
    ##################

    if args.temp_num is None and args.temp_list is None:
        ladder_num = tools.compute_ladder_num(
                f"{OUT_PATH}/{name}_explicit_equi",
                temperature,
                args.last_temp)
        ladder_num *= 2  # safety factor
        logger.info(f"Computed ladder number = {ladder_num}")
        temperatures = None
    elif args.temp_list is not None:
        ladder_num = len(args.temp_list)
        temperatures = args.temp_list
    else:
        temperatures = None
        ladder_num = args.temp_num

    dt = 4 * unit.femtosecond

    tot_steps = int(np.ceil(args.time * unit.nanoseconds / dt))
    save_step_dcd = 10000
    tempChangeInterval = int(args.temp_time / dt.in_units_of(unit.picosecond)._value)
    logger.info(f"Temperature change interval = {tempChangeInterval}")

    if args.log_time is not None:
        save_step_log = int(args.log_time / dt.in_units_of(unit.picosecond)._value)
    else:
        save_step_log = tempChangeInterval

    logger.info(f"Log save interval = {save_step_log}")

    temp_list = tools.compute_temperature_list(
        minTemperature=args.min_temp,
        maxTemperature=args.last_temp,
        numTemperatures=ladder_num)
    
    logger.info(f"Using temperatures : {', '.join([str(round(temp.in_units_of(unit.kelvin)._value, 2)) for temp in temp_list])}")
    logger.info(f"- Launch ST simulation {temp_list}")

    run_st(
        simulation,
        cif.topology,
        f"{OUT_PATH}/{name}_ST",
        tot_steps,
        dt=dt,
        temperatures=temp_list,
        save_step_dcd=save_step_dcd,
        save_step_log=save_step_log,
        tempChangeInterval=tempChangeInterval,
        )