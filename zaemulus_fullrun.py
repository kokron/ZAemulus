#Import everything you need
import numpy as np
import matplotlib.pyplot as plt
import time
from nbodykit.algorithms.fftpower import FFTPower
import gc
import pyfftw as fftw
import pyccl
import pandas as pd
import sys
from scipy.interpolate import interp1d
from nbodykit import mockmaker
from pmesh.pm import ParticleMesh
from nbodykit.source.mesh import ArrayMesh
from nbodykit.source.catalog import ArrayCatalog




#######################NOTE##################################
#Code is currently only set up to run on SLAC ki-ls nodes.###
#Do not attempt to run elsewhere, unspeakable dangers await.#
#############################################################

def delta_to_tidesq(delta_k, nmesh, lbox):
    #Assumes delta_k is a pyfftw fourier-transformed density contrast field
    #Computes the tidal tensor tau_ij = (k_i k_j/k^2  - delta_ij/3 )delta_k
    #Returns it as an nbodykit mesh
    
    
    kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    
    kx, ky, kz = np.meshgrid(kvals, kvals, kvalsr, indexing='ij')
    
    
    knorm = kx**2 + ky**2 + kz**2
    knorm[0][0][0] = 1
    klist = [[kx, kx], [kx, ky], [kx, kz], [ky, ky], [ky, kz], [kz, kz]]
    
    del kx, ky, kz
    gc.collect()
    
    
    #Compute the symmetric tide at every Fourier mode which we'll reshape later
    
    #Order is xx, xy, xz, yy, yz, zz
    
    
    jvec = [[0,0], [0,1], [0,2], [1,1], [1,2], [2,2]]
    tidesq = np.zeros(shape=(len(kvals), len(kvals), len(kvals)))

    #Transform each ij component individually, add to tide^2
    #s_ik s_kj = s_ii^2 + 2s_ij^2 for relevant i's, j's
    for i in range(len(klist)):
        fft_tide = np.array((klist[i][0]*klist[i][1]/knorm - diracdelta(jvec[i][0], jvec[i][1]) /3.) * (delta_k), dtype='complex64')
        real_tide = fftw.interfaces.numpy_fft.irfftn(fft_tide, axes=[0, 1, 2], threads=-1, auto_align_input=True)
        tidesq += real_tide**2
        if jvec[i][0] != jvec[i][1]:
            tidesq+= real_tide**2
            
    del real_tide, fft_tide
    gc.collect()

    return ArrayMesh(tidesq, BoxSize=lbox).to_real_field()

#Not using right now 
# def delta_to_gradsqdelta(delta_k, nmesh, lbox):
    
#     kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
#     kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    
#     kx, ky, kz = np.meshgrid(kvals, kvals, kvalsr, indexing='ij')
    
    
#     knorm = kx**2 + ky**2 + kz**2
#     knorm[0][0][0] = 1
    
#     ksqdelta = knorm*delta_k
    
#     ksqdelta = fftw.byte_align(ksqdelta, dtype='complex64')
    
#     gradsqdelta = fftw.interfaces.numpy_fft.irfftn(ksqdelta, axes=[0,1,2], threads=-1)

    
#     return ArrayMesh(gradsqdelta, BoxSize=lbox).to_real_field()
    

def diracdelta(i, j):
    if i == j:
        return 1
    else:
        return 0

def cosmoload(box=0):
    '''
    Load in the CCL cosmology from the Aemulus files
    Input:
        -box: integer number of Aemulus box
    Output:
        -cosmo: CCL cosmology object
    '''
    cosmofiles = pd.read_csv('~jderose/public_html/aemulus/phase1/cosmos.txt', sep=' ')

    box0cosmo = cosmofiles.iloc[box]

    cosmo = pyccl.Cosmology(Omega_b= box0cosmo['ombh2']/(box0cosmo['H0']/100)**2,
                         Omega_c = box0cosmo['omch2']/(box0cosmo['H0']/100)**2,
                         h = box0cosmo['H0']/100, n_s = box0cosmo['ns'], w0=box0cosmo['w0'], Neff=box0cosmo['Neff'],
                         sigma8 = box0cosmo['sigma8'])
    return cosmo

def ICpk(cosmo, z_ic):
    '''
    Compute the linear theory P(k) at the redshift of initial conditions
    Input:
        -cosmo: cosmological parameters
        -z_ic: initial conditions redshift
    Output:
        -fpk: power spectrum function that takes k in units of h/Mpc
    '''
    k = np.logspace(-5, 2.3, 1000)

    #CCL uses units if 1/Mpc not h/Mpc so we have to convert things
    pk = pyccl.linear_matter_power(cosmo, k*cosmo['h'], 1)*(cosmo['h'])**3
    D = pyccl.growth_factor(cosmo, 1./(1+z_ic))
    fpk = interp1d(k, pk*D**2)

    return fpk
def makemesh(Lbox, nmesh, seed, cosmo, z_ic):
    '''
    Make the IC linear mesh and displacements
    Input:
        -Lbox: size of the box in Mpc/h
        -nmesh: number of grid cells (or, equivalently, particles per dimension)
        -seed: random seed to initialize
        -cosmo: CCL cosmology for the box
        -z_ic: redshift at which box is being made
    Output:
        -delta: initial noiseless density field
        -disp: linear Lagrangian displacements at z_ic
    '''
    newmesh = ParticleMesh(BoxSize=Lbox, Nmesh=[nmesh, nmesh, nmesh])
    fpk = ICpk(cosmo, z_ic)
    delta, disp = mockmaker.gaussian_real_fields(newmesh, fpk, seed, compute_displacement=True)
    return delta, disp

def zaemulate(disp, z_ic, z_late, cosmo):
    '''
    Make a Zel'dovich-evolved catalog at late times. 
    Input:
        -disp: displacement field computed from 'makemesh'
        -z_ic: initial redshift
        -z_late: final redshift
        -cosmo: pyCCL cosmology used to compute growth factors
    Output:
        -flatgrid: catalog of positions for files
    '''

    #Get box size and particle number from the displacement
    Lbox = disp[0].BoxSize[0]
    nmesh = disp[0].Nmesh[0]

    #Coordinates to place down the IC particles
    coord = np.linspace(0, Lbox, nmesh+1)[:-1]
    xx, yy, zz = np.meshgrid(coord, coord, coord, indexing='ij')
    flatgrid = np.stack([xx,yy,zz])


    growthratio = pyccl.growth_factor(cosmo, 1./(1+z_late))/pyccl.growth_factor(cosmo, 1./(1+z_ic))

    #ZAemulate them to z_late
    for i in range(3):
        flatgrid[i] += disp[i]*growthratio
    flatgrid = flatgrid%Lbox

    #Reshape to catalog
    flatgrid = flatgrid.reshape(3, nmesh**3)

    return flatgrid

def cat_to_mesh(cat, Lbox, Nmesh):
    '''
    Convert the late-time particle catalog to a friendly neighbourhood mesh
    Input:
        -cat: format is [Npos, Nparticles]
        -Lbox: units are Mpc/h
        -Nmesh: size of mesh you want to deposit particles in
    Output:
        -mesh
    '''
    nbodycat = np.empty(nmesh**3, dtype=[('Position', ('f8', 3))])
    nbodycat['Position'] = cat.T

    nbkcat = ArrayCatalog(nbodycat, Nmesh=nmesh, BoxSize=Lbox)
    mesh = nbkcat.to_mesh(Nmesh=nmesh, BoxSize=Lbox)

    field = (mesh.paint(mode='real')-1)
    return field

def make_component_weights(icdeltalin, cosmo, z_ic, z_late):
    '''
    Make the component fields so you get weights later on.
    Input:
        icdeltalin: **NOISELESS** linear fields of your simulation
        growthratio: ratio of growthfactors between z_late and z_ic
    Output:
        linfield: linear noiseless density
        delta2: squared density field
        s2: tidal field squared
        gradsqdelta: laplacian of the density
    Notes:
        gradsqdelta currently not implemented
    '''
    Lbox = icdeltalin.BoxSize[0]
    nmesh = icdeltalin.Nmesh[0]

    growthratio = pyccl.growth_factor(cosmo, 1./(1+z_late))/pyccl.growth_factor(cosmo, 1./(1+z_ic))

    delta = growthratio*icdeltalin

    #Start with tidal field due to memory requirements
    linfield = fftw.byte_align(icdeltalin.value, dtype='float32')


    field_fft = fftw.interfaces.numpy_fft.rfftn(linfield, threads=-1)

    linfield = ArrayMesh(linfield, BoxSize=Lbox).to_real_field()

    tidesq = delta_to_tidesq(field_fft, nmesh=nmesh, lbox=Lbox)
    s2 = tidesq - tidesq.cmean()

    del field_fft
    gc.collect()

    #Make the squared density field
    sqfield = delta**2

    delta2 = sqfield -  sqfield.cmean()

    return linfield, delta2, s2

def advected_field(cat, field_late, weight, nmesh):
    '''
    Get the advected component field with the weights given by the 'weight' column.

    Input:
        -cat: catalog of galaxy positions, assumed to have dimensions [3, N] right now
        -field_late: late-time density field (the delta_1 component) which is needed to 
            prevent an nbodykit bug when placing down weights
        -weight: the early time component field used for the weights. See Note about 1-1 mapping
        -nmesh: Nmesh for the late-time advected field. Can be different from the grid value of the component field.
    Output:
        -field_complate: late-time, advected component field for the given input weights.

    Notes: In ZAemulus it's assumed that there's a clear 1-1 mapping between positions
    and particle since this comes from reshaping the grids. This has to be changed for
    a more realistic particle catalog.
    '''
    Lbox = field_late.BoxSize[0]

    nbodycat = np.empty(nmesh**3, dtype=[('Position', ('f8', 3)), ('Weight', 'f8')])
    nbodycat['Position'] = cat.T
    #Collapse the weight grids into catalog shape.
    #Need to change this for future runs.
    #+1 is needed if subtracting from field_late
    nbodycat['Weight'] = weight.value.reshape(nmesh**3)+1


    nbkcat = ArrayCatalog(nbodycat, Nmesh=nmesh, BoxSize=Lbox)
    mesh = nbkcat.to_mesh(Nmesh=nmesh, BoxSize=Lbox, weight='Weight')

    #Paint and subtract from late field to avoid annoying nbodykit bugs
    field_component_late = (mesh.paint(mode='real')-1)
    field_component_late -= field_late

    return field_component_late

def triangle_plot(filename, cleftname, field_array, z_late):
    '''
    Makes a giant triangle plot comparing the late-time fields to a CLEFT prediction, making the measurements required
    assuming the input fields are correctly structured. 

    Input:
        -filename: directory where you want the triangle plot to be saved
        -cleftname: directory where you will pull the CLEFT predictions for comparison
        -field_array: list of the component late-time fields we will use to make comparisons.

    Output:
        -pdf plot at filename
        -Component spectra 

    Note: field_array assumes the following ordering (See Modi, Chen, White 2020 for definitions):
        delta_1, delta_delta_L, delta_delta^2, delta_s^2
    '''
    field_late, field_linlate, field_sqlate, tide_sqlate = field_array

    Lbox = field_late.BoxSize[0]
    nmesh = field_late.Nmesh[0]
    pk = np.loadtxt(cleftname)
    fig, axes = plt.subplots(figsize=(10, 10), sharex=True, ncols=4, nrows=4)
    for i in range(4):
        for j in range(4):
            if i<j:
                axes[i, j].axis('off')

    field_dict = {'1': field_late, r'$\delta_L$': field_linlate, r'$\delta^2$': field_sqlate, r'$s^2$': tide_sqlate}


    b1sq = FFTPower(field_dict[r'$\delta_L$'], '1d', second=field_dict[r'$\delta_L$'], BoxSize=Lbox, Nmesh=nmesh)
    b1power = b1sq.power['power'].real

    axes[0, 0].loglog(b1sq.power['k'], b1power, label=r'Nbody', lw=2)
    axes[0, 0].loglog(pk[0], pk[5], lw=2, label=r'CLEFT')
    axes[0, 0].set_title(r'$b_1^2$ component')
    axes[0, 0].set_xlim(pk[0][1], pk[0][-1])

    b2sq = FFTPower(field_dict[r'$\delta^2$'], '1d', second=field_dict[r'$\delta^2$'], BoxSize=Lbox, Nmesh=nmesh)
    b2power = b2sq.power['power'].real
    axes[1,0].loglog(b2sq.power['k'], b2power, label=r'Nbody', lw=2)
    axes[1,0].loglog(pk[0], pk[8], lw=2, label=r'CLEFT')
    axes[1,0].set_title(r'$b_2^2$ component')

    b2b1 = FFTPower(field_dict[r'$\delta_L$'], '1d', second=field_dict[r'$\delta^2$'], BoxSize=Lbox, Nmesh=nmesh)
    b2b1power = b2b1.power['power'].real

    axes[1,1].loglog(b2b1.power['k'], b2b1power, label=r'Nbody', lw=2)
    axes[1,1].loglog(pk[0], pk[7], lw=2, label=r'CLEFT')
    axes[1,1].set_title('$b_2b_1$ component')

    fig.suptitle(r'CLEFT v.s. ZAemulus at $z=%s$, $N_{mesh} = %s$'%(z_late,nmesh), y=0.92)


    bs2 = FFTPower(field_dict[r'$s^2$'], '1d', second=field_dict[r'$s^2$'], BoxSize=Lbox, Nmesh=nmesh)
    bs2power = bs2.power['power'].real
    axes[2,0].loglog(bs2.power['k'], bs2power, label=r'Nbody', lw=2)
    axes[2,0].loglog(pk[0], pk[12], lw=2, label=r'CLEFT')
    axes[2,0].set_title(r'$b_s^2$ component')

    bsb1 = FFTPower(field_dict[r'$s^2$'], '1d', second=field_dict[r'$\delta_L$'], BoxSize=Lbox, Nmesh=nmesh)
    bsb1power = bsb1.power['power'].real

    axes[2,1].loglog(bsb1.power['k'], np.abs(bsb1power), label=r'Nbody', lw=2)
    axes[2,1].loglog(pk[0], np.abs(pk[10]), lw=2, label=r'CLEFT')
    axes[2,1].set_title('$b_sb_1$ component')

    bsb2 = FFTPower(field_dict[r'$s^2$'], '1d', second=field_dict[r'$\delta^2$'], BoxSize=Lbox, Nmesh=nmesh)
    bsb2power = bsb2.power['power'].real

    axes[2,2].loglog(bsb2.power['k'], np.abs(bsb2power), label=r'Nbody', lw=2)
    axes[2,2].loglog(pk[0], np.abs(pk[11]), lw=2, label=r'CLEFT')
    axes[2,2].set_title('$b_sb_2$ component')


    nonlinsq = FFTPower(field_dict['1'], '1d', second=field_dict['1'], BoxSize=Lbox, Nmesh=nmesh)
    nonlinsqpower = nonlinsq.power['power'].real
    axes[3,0].loglog(nonlinsq.power['k'], np.abs(nonlinsqpower), label=r'Nbody', lw=2)
    axes[3,0].loglog(pk[0], pk[1]+pk[2]+pk[3], label=r'CLEFT')
    axes[3,0].set_title('$(1,1)$ component')


    nonlinb1 = FFTPower(field_dict['1'], '1d', second=field_dict[r'$\delta_L$'], BoxSize=Lbox, Nmesh=nmesh)
    nonlinb1power = nonlinb1.power['power'].real
    axes[3,1].loglog(nonlinb1.power['k'], np.abs(nonlinb1power), label=r'Nbody', lw=2)
    axes[3,1].loglog(pk[0], pk[4], label=r'CLEFT')
    axes[3,1].set_title('$(1,b_1)$ component')


    nonlinbs = FFTPower(field_dict['1'], '1d', second=field_dict[r'$s^2$'], BoxSize=Lbox, Nmesh=nmesh)
    nonlinbspower = nonlinbs.power['power'].real
    axes[3,2].loglog(nonlinbs.power['k'], np.abs(nonlinbspower), label=r'Nbody', lw=2)
    axes[3,2].loglog(pk[0], pk[9], label=r'CLEFT')
    axes[3,2].set_title('$(1,b_s)$ component')

    nonlinb2 = FFTPower(field_dict['1'], '1d', second=field_dict[r'$\delta^2$'], BoxSize=Lbox, Nmesh=nmesh)
    nonlinb2power = nonlinb2.power['power'].real
    axes[3,3].loglog(nonlinb2.power['k'], np.abs(nonlinb2power), label=r'Nbody', lw=2)
    axes[3,3].loglog(pk[0], pk[6], label=r'CLEFT')
    axes[3,3].set_title('$(1,b_2)$ component')

    axes[0,0].legend()
    axes[0,0].legend()  

    fig.savefig(filename, dpi=300, format='pdf')
    #Note we're assuming they all end up being measured at the same k (which is fine?)
    return [nonlinb2.power['k'], b1power, b2power, b2b1power, bs2power, bsb1power, bsb2power, nonlinsqpower, nonlinb1power, nonlinbspower, nonlinb2power]
#Give arguments Lbox, nmesh, seed

if __name__ == '__main__':

    Lbox = float(sys.argv[1])

    nmesh = int(sys.argv[2])

    seed = int(sys.argv[3])

    nbox = int(sys.argv[4])

    start_time = time.time()

    z_ic = 49
    z_late = 1
    #Make the catalog
    cosmo = cosmoload(nbox)
    icdelta, icdisp = makemesh(Lbox, nmesh, seed, cosmo, z_ic)
    print('Initial field made. Took ', time.time() - start_time)
    #Go to late times
    latecat = zaemulate(icdisp, z_ic, z_late, cosmo)

    #Get lagrangian weights
    linfield, delta2, s2 = make_component_weights(icdelta, cosmo, z_ic, z_late)
    print('Lagrangian weights built. Took ', time.time() - start_time)
    #late time nonlinear field
    field_late = cat_to_mesh(latecat, Lbox, nmesh)

    #Advect fields to late times
    print('Advecting particles and making plots')
    latelin = advected_field(latecat, field_late, linfield, nmesh)
    lated2 = advected_field(latecat, field_late, delta2, nmesh)
    lates2 = advected_field(latecat, field_late, s2, nmesh)

    field_arr = [field_late, latelin, lated2, lates2]

    triangle_plot('/u/ki/nkokron/Projects/ZAemulus/testfile.png', '/u/ki/nkokron/notebooks/ptbias_emu/lpt_components_z1.0.dat', field_arr, z_late)
    print('Done with run. Took', time.time() - start_time)
