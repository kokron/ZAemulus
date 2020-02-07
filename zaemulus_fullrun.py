#Import everything you need
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
from nbodykit.algorithms.fftpower import FFTPower
import gc
from matplotlib import cm
import pyfftw as fftw
import pyccl
import pandas as pd
import struct
from collections import namedtuple
from scipy.interpolate import interp1d
from nbodykit import mockmaker
from pmesh.pm import ParticleMesh
from nbodykit.source.mesh import ArrayMesh
from nbodykit.source.catalog import ArrayCatalog
from mpl_toolkits.axes_grid1 import ImageGrid




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
def delta_to_gradsqdelta(delta_k, nmesh, lbox):
    
    kvals = np.fft.fftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    kvalsr = np.fft.rfftfreq(nmesh)*(2*np.pi*nmesh)/lbox
    
    kx, ky, kz = np.meshgrid(kvals, kvals, kvalsr, indexing='ij')
    
    
    knorm = kx**2 + ky**2 + kz**2
    knorm[0][0][0] = 1
    
    ksqdelta = knorm*delta_k
    
    ksqdelta = fftw.byte_align(ksqdelta, dtype='complex64')
    
    gradsqdelta = fftw.interfaces.numpy_fft.irfftn(ksqdelta, axes=[0,1,2], threads=-1)

    
    return ArrayMesh(gradsqdelta, BoxSize=lbox).to_real_field()
    

def diracdelta(i, j):
    if i == j:
        return 1
    else:
        return 0

def cosmoload(box=0):
	'''
	Load in the CCL cosmology from the Aemulus files
	'''
	cosmofiles = pd.read_csv('~jderose/public_html/aemulus/phase1/cosmos.txt', sep=' ')

	box0cosmo = cosmofiles.iloc[i]

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
	pk = pyccl.linear_matter_power(cosmo, k*box0cosmo['H0']/100, 1)*(box0cosmo['H0']/100)**3
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

def zaemulate(disp, z_ic, z_late):
	'''
	Make a Zel'dovich-evolved catalog at late times. 
	Input:
		-disp: displacement field computed from 'makemesh'
		-z_ic: initial redshift
		-z_late: final redshift
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

def make_component_weights(icdeltalin, growthratio):
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

	Lbox = icdeltalin[0].BoxSize[0]
	nmesh = icdeltalin[0].Nmesh[0]

	delta = growthratio*icdeltalin

	#Start with tidal field due to memory requirements
	linfield = fftw.byte_align(icdeltalin.value, dtype='float32')


	field_fft = fftw.interfaces.numpy_fft.rfftn(linfield, threads=-1)

	linfield = ArrayMesh(linfield, BoxSize=Lbox).to_real_field()

	tidesq = delta_to_tidesq(field_fft, nmesh=nmesh, lbox=Lbox)
	s2 -= tidesq.cmean()

	del field_fft
	gc.collect()

	#Make the squared density field
	sqfield = delta**2

	delta2 -= sqfield.cmean()

	return linfield, delta2, s2

def advected_field(cat, weight, nmesh, Lbox):
	

#Give arguments Lbox, nmesh, seed
Lbox = sys.argv[1]

nmesh = sys.argv[2]

seed = sys.argv[3]