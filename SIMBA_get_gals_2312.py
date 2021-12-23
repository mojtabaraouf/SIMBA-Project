#date 11.03.2021
#The ram pressure striping estimate for the shell between the r_vir: halo.radii.virial and 2r_vir
#the restoring force calculated for galaxies up to 2 Re:  g.radii.totalhalfmass.
#The truncation radii estimete when the ram pressure and restoring force are equal with 0.5dex difference

import matplotlib
# import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage
import numpy as np
from numpy import ma
from matplotlib import colors, ticker, cm
from matplotlib.mlab import bivariate_normal
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from mpl_toolkits import mplot3d
import h5py
import numpy as np
import os
import caesar
from astropy import constants as const
from readgadget import *

# Functions
#----------------------------------------------------------------------------------------------------
def get_r(mass,pos,r):
    # positions need to be in kpc
    mtot = np.sum(mass)
    pos_new = np.zeros_like(pos)

    x0 = np.sum(pos[:,0]*mass)/np.sum(mass)
    y0 = np.sum(pos[:,1]*mass)/np.sum(mass)
    z0 = np.sum(pos[:,2]*mass)/np.sum(mass)
    pos_new[:,0] = pos[:,0] - x0
    pos_new[:,1] = pos[:,1] - y0
    pos_new[:,2] = pos[:,2] - z0

    mr50 = 0.
    dr = 10              # in kpc
    i = 0
    while (mr50 < mtot*r/100.):
         i += 1
         mask = ((pos_new[:,0]**2 + pos_new[:,1]**2 + pos_new[:,2]**2) <= (dr*i)**2)
         mr50 = np.sum(mass[mask])
    return dr*i,x0,y0,z0

def get_cm(mass,pos):
    # positions need to be in kpc
    mtot = np.sum(mass)
    pos_new = np.zeros_like(pos)

    x0 = np.sum(pos[:,0]*mass)/np.sum(mass)
    y0 = np.sum(pos[:,1]*mass)/np.sum(mass)
    z0 = np.sum(pos[:,2]*mass)/np.sum(mass)

    return x0,y0,z0
def LoadMetal(gmet,MetIndex):   # returns name, solar abundance, and solar-scaled metallicity array for given metal index
    MetName = ['H','He','C','N','O','Ne','Mg','Si','S','Ca','Fe']
    SolarAbundances=[0.0134, 0.2485, 2.38e-3, 0.70e-3, 5.79e-3, 1.26e-3, 7.14e-4, 6.71e-3, 3.12e-4, 0.65e-4, 1.31e-3]
    Zmet = np.asarray([i[MetIndex] for i in gmet])
    #Zmet /= SolarAbundances[MetIndex]
    return MetName[MetIndex],SolarAbundances[MetIndex],Zmet
#----------------------------------------------------------------------------------------------------
for snap in range(151,150,-1): #in range(150,151):
  # if ((snap != 109)&(snap != 113)&(snap != 105)&(snap != 78)&(snap != 151)&(snap != 126)):
  if ((snap == 151)|(snap == 105)|(snap == 78)):
    #print snap
    #snap = '151'   m50n512_057.hdf5 # z=0: 151; z=0.5: 125; z=1: 105; z=2: 078
    output = './result/SIMBA_2312_get_gal/Gals_fof6d_m50n512_'+str(snap).zfill(3)+'_2vir_T.asc'
    profile_frest = './result/SIMBA_2312_get_gal/fresor_m50n512_'+str(snap).zfill(3)+'.asc'

    # f = open(profile_frest, 'w+')
    # f.write('# galID r df_ramp frestor Re Rh200 Mh200 cent df_ramp_dm df_ramp_g frestor_h frestor_g frestor_s densi massdmi massgasi massstari'+"\n")

#IN URSA
    # snapfile = '/home/rad/data/m50n512/s50fof6d/snap_m50n512_'+str(snap).zfill(3)+'.hdf5'
    # caesarfile = '/home/rad/data/m50n512/s50j7k/Groups/m50n512_'+str(snap).zfill(3)+'.hdf5'
#In KASI
    snapfile = './SIMBA/SIMBA/snap_m50n512_'+str(snap).zfill(3)+'.hdf5'
    # caesarfile = '/media/mraouf/mraouf2/Groups/m50n512_'+str(snap).zfill(3)+'.hdf5'
    caesarfile = './SIMBA/Groups/m50n512_'+str(snap).zfill(3)+'.hdf5'

# load in input file
    #sim = caesar.load(caesarfile,LoadHalo=False)  # load without halos
    sim = caesar.load(caesarfile)

    ngal = sim.ngalaxies
    #print 'ngal:',ngal
    bsize  = sim.simulation.boxsize.to('kpccm').d
    #print 'boxsize',bsize

    redshift = np.round(sim.simulation.redshift,decimals=2)
    print 'redshift:',redshift

    h = sim.simulation.hubble_constant
    #print 'h:',h
    kpc2km = 3.086*10**(16.)     # conversion from kpc to km

# relevant galaxy info
    galID = np.asarray([i.GroupID for i in sim.galaxies]).astype(np.int)
    galMstar = np.asarray([i.masses['stellar'] for i in sim.galaxies])
    galsfr = np.asarray([i.sfr.to('Msun/yr').d for i in sim.galaxies])
    met = np.asarray([i.metallicities['sfr_weighted'] for i in sim.galaxies])
    mh = np.asarray([i.halo.masses['total'] for i in sim.galaxies])
    galVel = np.asarray([i.vel.to('kmcm/s').d for i in sim.galaxies])
    galPos = np.asarray([i.pos.to('kpccm').d for i in sim.galaxies])   # in kpc
   # mdot = np.asarray([i.bhmdot for i in sim.galaxies])
   # mbh = np.asarray([i.masses['bh'] for i in sim.galaxies])

    galcen_orig = np.array([i.central for i in sim.galaxies]) #.astype(np.bool)
    galcen = np.zeros(len(galcen_orig)).astype(np.int)
    galcen[galcen_orig] = 1
    #print 'min-max galcen:',np.min(galcen),np.max(galcen)

    galHImass = np.asarray([i.masses['HI'] for i in sim.galaxies])
    galH2mass = np.asarray([i.masses['H2'] for i in sim.galaxies])

    galPos = np.asarray([i.pos.to('kpccm').d for i in sim.galaxies])   # in kpc
    galVel = np.asarray([i.vel.to('kmcm/s').d for i in sim.galaxies])

    galPos = galPos/1000.  #Mpc/h
    gals = np.asarray([i for i in sim.galaxies])
    cens = np.asarray([i for i in sim.galaxies if i.central==True])
    sats = np.asarray([i for i in sim.galaxies if i.central==False])


#halo properties
    mhlim = 11
    halos = np.asarray([i.halo for i in sim.galaxies])# if i.halo.masses['total']>10**mhlim])
    haloid = np.asarray([i.halo.GroupID for i in gals if i.halo.masses['total']>10**mhlim])
    mh = np.asarray([i.masses['total'] for i in halos])
    m500 = np.log10(0.7*mh)	# 0.7 is approximate conversion from virial mass to M500
    # Temp_vir = np.asarray([i.temperatures['virial'] for i in sim.galaxies])

    Zsol = 1.31e-3
    UnitLength_in_cm=3.08568e+21	#WATCH OUT: kpc/h
    UnitMass_in_g = 1.989e+33		#WATCH OUT: Msun
    UnitDensity_in_cgs = UnitMass_in_g / (UnitLength_in_cm**3)
    Thot = 3.e5	# min gas temperature to use in computing X-ray
    nskip = 1	# choose every nskip'th particle (for debugging)

#---------------------------------------------------------------------------------------
#reads snap particles
#---------------------------------------------------------------------------------------

    # read in the particle information we need, mass, positions and velocity
    pmass = readsnap(snapfile,'mass','star',units=1,suppress=1)/h  # note the h^-1 from Gadget units

    ppos = readsnap(snapfile,'pos','star',units=0,suppress=1)/h      #/h/1000.-> in Mpc; otherwise in kpc comoving: max 50Mpc/h
    pvel = readsnap(snapfile,'vel','star',units=0,suppress=1)  # in km/s
    pmetarray = readsnap(snapfile,'Metallicity','star',units=1,suppress=1)
    tform = readsnap(snapfile,'age','star',units=1,suppress=1)  # expansion factor of formation
    Zname1,ZSolar1,Zmet1 = LoadMetal(pmetarray,6)
    Zname2,ZSolar2,Zmet2 = LoadMetal(pmetarray,10)
    # print 'Read ',snapfile,' z= ',redshift,' h= ',h,' Metals',Zname1,Zname2

    pmass_dm = readsnap(snapfile,'mass','dm',units=1,suppress=1)/h
    ppos_dm = readsnap(snapfile,'pos','dm',units=0,suppress=1)/h      #/h/1000.-> in Mpc; otherwise in kpc comoving: max 50Mpc/h
    pvel_dm = readsnap(snapfile,'vel','dm',units=0,suppress=1)  # in km/s

    pmass_gas = readsnap(snapfile,'mass','gas',units=1,suppress=1)/h  # note the h^-1 from Gadget units
    pvel_gas = readsnap(snapfile,'vel','gas',units=0,suppress=1,nth=nskip)  # in km/s
    pdens_gas = readsnap(snapfile,'Density','gas',units=1,suppress=1,nth=nskip)# * UnitDensity_in_cgs  # in
    ppos_gas = readsnap(snapfile,'pos','gas',units=0,suppress=1,nth=nskip)/h      #/h/1000.-> in Mpc; otherwise in kpc comoving: max 50Mpc/h
    gnh = readsnap(snapfile,'rho','gas',units=1,suppress=1,nth=nskip) *h*h*0.76 #/1.673e-24	# number density in H atoms/cm^3  atom H = 1.67 * 1e-24 g
    gnhyd = readsnap(snapfile,'nh','gas',units=1,suppress=1)	# neutral hydrogen fraction (0-1)
    gne = readsnap(snapfile,'ne','gas',units=1,suppress=1)	# electron number density, relative to H density (0-1.17 or so)
    gu = readsnap(snapfile,'u','gas',units=1,suppress=1,nth=nskip)		# temperature in K
    gdelay = readsnap(snapfile,'delaytime','gas',units=1,suppress=1,nth=nskip)	# delay time
    gsfr = readsnap(snapfile,'sfr','gas',units=1,suppress=1,nth=nskip)	# SFR in Mo/yr
    gnwind = readsnap(snapfile,'NWindLaunches','gas',suppress=1)	# Number of wind launches (AGN launches tracked in 1000's place)
    h1 = readsnap(snapfile,'NeutralHydrogenAbundance','gas',units=0,suppress=1)  # in km/s

    ghaloid = readsnap(snapfile,'HaloID','gas',suppress=1)	# Halo ID of particle; 0 if not in halo
    ghaloid = np.array(ghaloid).astype(int)		# for some reason Halo ID's are stored as floats!  set to ints
    # gas_select = ((gu>Thot)&(gdelay==0)&(gsfr==0))  # select gas for which to add up X-ray quantities
    gas_select = ((gu<1e10))

    gas_select_c = ((gu<Thot))
    gas_select_h = ((gu>Thot)&(gdelay==0)&(gsfr==0))
    gas_select_all = ((gu>1))  # select gas allv for outflow and inflow gas

    sfselect = ((gsfr>0))#  SF gas
    wdselect = ((gdelay>0))#  wind gas
    jetselect = ((gnwind>=1000))#  jet gas
    gselect = ((gu>1e9)&(gnh>0.00001))


    vel_wrt2 = [];dens_g = [];ram_p = [];nsf = []; nwd = []; njet = [];ncgas = []
    outflow= []; inflow= []; all_gas = []; Temp = []; M_acc_cold = [];M_acc_hot = []
    vel_wrt2_c = [];dens_g_c = [];ram_p_c=[];vel_wrt2_h = [];dens_g_h = [];ram_p_h=[]
    dens_r= []; mass_r= []; r_size= []; f_restor_vir= [];r_trunc = [];
    f_restor_vir_star=[];f_restor_re=[];f_restor_re_dm=[];f_restor_re_gas=[];f_restor_re_star=[];mass_r_re=[];dens_r_re=[];
    f_restor_trunc=[];   f_restor_vir_dm=[]; f_restor_vir_gas=[];   hc_dist_center=[];hc_r200=[];redshift1 = []; snap1 = [];r_200 = [];M_200 = []
    ii = 0
#---------------------------------------------------------------------------------------
#galaxy info
#---------------------------------------------------------------------------------------

    for g in gals:
       #if  g.GroupID == 638:
        # print 'ii & halo count:',ii,len(halos)
        velgas = np.array([pvel_gas[k] for k in g.halo.glist  if (gas_select[k])])
        mgas = np.array([pmass_gas[k] for k in g.halo.glist  if (gas_select[k])])
        posgas = np.array([ppos_gas[k] for k in g.halo.glist  if (gas_select[k])])
        densgas = np.array([gnh[k] for k in g.halo.glist  if (gas_select[k])])

        mdm = np.array([pmass_dm[k] for k in g.halo.dmlist])
        posdm = np.array([ppos_dm[k] for k in g.halo.dmlist])
        veldm = np.array([pvel_dm[k] for k in g.halo.dmlist])

        mstar = np.array([pmass[k] for k in g.halo.slist])
        posstar = np.array([ppos[k] for k in g.halo.slist])
        vels = np.array([pvel[k] for k in g.halo.slist])

        sfgnh = np.array([gnh[k] for k in g.halo.glist  if (sfselect[k])])
        wdgnh = np.array([gnh[k] for k in g.halo.glist  if (wdselect[k])])
        jetgnh = np.array([gnh[k] for k in g.halo.glist  if (jetselect[k])])
        # gasgnh = np.array([gnh[k] for k in g.glist  if (gselect[k])])
        gasTemp = np.array([gu[k] for k in g.halo.glist  if (gas_select[k])])

        # All gas
        velgas_all = np.array([pvel_gas[k] for k in g.halo.glist  if (gas_select_all[k])])
        mgas_all = np.array([pmass_gas[k] for k in g.halo.glist  if (gas_select_all[k])])
        posgas_all = np.array([ppos_gas[k] for k in g.halo.glist  if (gas_select_all[k])])

        # cold gas
        velgas_c = np.array([pvel_gas[k] for k in g.halo.glist  if (gas_select_c[k])])
        mgas_c = np.array([pmass_gas[k] for k in g.halo.glist  if (gas_select_c[k])])
        posgas_c = np.array([ppos_gas[k] for k in g.halo.glist  if (gas_select_c[k])])
        densgas_c = np.array([gnh[k] for k in g.halo.glist  if (gas_select_c[k])])

        # hot gas
        velgas_h = np.array([pvel_gas[k] for k in g.halo.glist  if (gas_select_h[k])])
        mgas_h = np.array([pmass_gas[k] for k in g.halo.glist  if (gas_select_h[k])])
        posgas_h = np.array([ppos_gas[k] for k in g.halo.glist  if (gas_select_h[k])])
        densgas_h = np.array([gnh[k] for k in g.halo.glist  if (gas_select_h[k])])

        if ((len(mgas) > 0)):
            print 'snap & ii & halo count:',snap,ii,len(halos)

            galPosx =  g.pos.d[0]
            galPosy =  g.pos.d[1]
            galPosz =  g.pos.d[2]
            hc_Posx =  g.halo.central_galaxy.pos.d[0]
            hc_Posy =  g.halo.central_galaxy.pos.d[1]
            hc_Posz =  g.halo.central_galaxy.pos.d[2]

            gal_dist_center = (np.sqrt((hc_Posx - galPosx)**2 + (hc_Posy - galPosy)**2 + (hc_Posz - galPosz)**2))
            hcr200 = g.halo.radii['r200c'].d

            vx = g.vel.d[0]
            vy = g.vel.d[1]
            vz = g.vel.d[2]
            galMass = g.masses['stellar'].d

            if g.radii['total_half_mass'] > 1.:
                radii = g.radii['total_half_mass'].d
                #radii_h = g.halo.radii['virial'].d
            else:
                # break
                radii =  1.0
            #if galcen[ii] == 0:
            if 'r200c' in g.radii:

                radii_h = g.radii['r200c'].d

            else:
                radii_h = g.radii['total'].d


            #else:

                    #radii_h =g.halo.radii['r200c'].d
     #       print 'galPos:',galPosx,galPosy,galPosz,'rvir:', radii
            #Ram pressure 1
            rsize_h = radii_h
            Grav = 4.3 * 1e-6# Km2 kpc Msun-1 s-2
            Pi = 3.14
            kpc_cm = 3.086e+21
            M_sun = 1.989e+33

            rx = posgas[:,0] - galPosx
            ry = posgas[:,1] - galPosy
            rz = posgas[:,2] - galPosz
            ro = np.sqrt((rx)**2 + (ry)**2)
            teta = (np.arctan(ry/rx))
            x1 =galPosx;x2 = posgas[:,0]
            y1 =galPosy;y2 = posgas[:,1]
            z1 =galPosz;z2 = posgas[:,2]
            r1 = np.sqrt((x1)**2 + (y1)**2)
            r2 = np.sqrt((x2)**2 + (y2)**2)
            teta1 = (np.arctan(y1/x1))
            teta2 = (np.arctan(y2/x2))
            Dist_project = np.sqrt(rx**2+ry**2)
            Dist_cylind = np.sqrt((r1)**2 + (r2)**2 - 2*r1*r2*np.cos(teta1 - teta2))
            Dist_sphere = (rx)**2 + (ry)**2 + (rz)**2
            phi1 = (np.arccos(z1/np.sqrt(x1**2+y1**2+z1**2)))
            phi2 = (np.arccos(z2/np.sqrt(x2**2+y2**2+z2**2)))
            phi = (np.arccos(rz/np.sqrt(rx**2+ry**2+rz**2)))
            Vesc = np.sqrt(2. * Grav * g.masses['total'].d / rsize_h)


            mask = ( (np.sqrt( Dist_sphere)> radii_h) &(np.sqrt( Dist_sphere) <= 2.0*radii_h))
            gasVelx = np.sum(mgas[mask]*velgas[:,0][mask])/np.sum(mgas[mask])
            gasVely = np.sum(mgas[mask]*velgas[:,1][mask])/np.sum(mgas[mask])
            gasVelz = np.sum(mgas[mask]*velgas[:,2][mask])/np.sum(mgas[mask])
            # vx_gas = velgas[:,0][mask] - gasVelx
            # vy_gas = velgas[:,1][mask] - gasVely
            # vz_gas = velgas[:,2][mask] - gasVelz
            #dens1 = np.sum(mgas[mask]*densgas[mask])/np.sum(mgas[mask])
            #mask_den = ( (np.sqrt( Dist_sphere)> radii_h) &(np.sqrt( Dist_sphere) <= 2.0*radii_h)& (np.sqrt((vx_gas)**2 + (vy_gas)**2 + (vz_gas)**2)<= Vesc))
            dens = np.mean(densgas[mask])
            vel_wrt_gas2 = (vx - gasVelx)**2 + (vy - gasVely)**2 + (vz - gasVelz)**2
            ram = (dens * vel_wrt_gas2)


            # For estimate Restoring gas at virial radii
            rx_s = posstar[:,0] - galPosx
            ry_s = posstar[:,1] - galPosy
            rz_s = posstar[:,2] - galPosz
            #ro_s = np.sqrt((rx_s)**2 + (ry_s)**2)
            #teta_s = (np.arctan(ry_s/rx_s))
            x2_s = posstar[:,0]
            y2_s = posstar[:,1]
            z2_s = posstar[:,2]
            r2_s = np.sqrt((x2_s)**2 + (y2_s)**2)
            teta2_s = (np.arctan(y2_s/x2_s))
            #Dist_cylind_s = np.sqrt((r1)**2 + (r2_s)**2 - 2*r1*r2_s*np.cos(teta1 - teta2_s))
            Dist_sphere_s = (rx_s)**2 + (ry_s)**2 + (rz_s)**2

            rx_dm = posdm[:,0] - galPosx
            ry_dm = posdm[:,1] - galPosy
            rz_dm = posdm[:,2] - galPosz
            x2_dm = posdm[:,0]
            y2_dm = posdm[:,1]
            z2_dm = posdm[:,2]
            r2_dm = np.sqrt((x2_dm)**2 + (y2_dm)**2)
            teta2_dm = (np.arctan(y2_dm/x2_dm))
            #Dist_cylind_dm = np.sqrt((r1)**2 + (r2_dm)**2 - 2*r1*r2_dm*np.cos(teta1 - teta2_dm))
            Dist_sphere_dm = (rx_dm)**2 + (ry_dm)**2 + (rz_dm)**2

            mask_r =  ((np.sqrt( Dist_sphere) <= radii_h))
            gVelx = np.sum(mgas[mask_r]*velgas[:,0][mask_r])/np.sum(mgas[mask_r])
            gVely = np.sum(mgas[mask_r]*velgas[:,1][mask_r])/np.sum(mgas[mask_r])
            gVelz = np.sum(mgas[mask_r]*velgas[:,2][mask_r])/np.sum(mgas[mask_r])


            vx_g = velgas[:,0] - gVelx
            vy_g = velgas[:,1] - gVely
            vz_g = velgas[:,2] - gVelz
            mask_r =  ((np.sqrt( Dist_sphere_dm) <= radii_h))
            dmVelx = np.sum(mdm[mask_r]*veldm[:,0][mask_r])/np.sum(mdm[mask_r])
            dmVely = np.sum(mdm[mask_r]*veldm[:,1][mask_r])/np.sum(mdm[mask_r])
            dmVelz = np.sum(mdm[mask_r]*veldm[:,2][mask_r])/np.sum(mdm[mask_r])

            vx_dm = veldm[:,0] - dmVelx
            vy_dm = veldm[:,1] - dmVely
            vz_dm = veldm[:,2] - dmVelz

            mask_r =  ((np.sqrt( Dist_sphere_s) <= radii_h))
            starVelx = np.sum(mstar[mask_r]*vels[:,0][mask_r])/np.sum(mstar[mask_r])
            starVely = np.sum(mstar[mask_r]*vels[:,1][mask_r])/np.sum(mstar[mask_r])
            starVelz = np.sum(mstar[mask_r]*vels[:,2][mask_r])/np.sum(mstar[mask_r])

            vx_s = vels[:,0] - starVelx
            vy_s = vels[:,1] - starVely
            vz_s = vels[:,2] - starVelz

            #& ( np.sqrt((vx_g)**2 + (vy_g)**2 + (vz_g)**2)<= Vesc)
            # Virial halo restoring force
            r_1 = rsize_h - 1.0
            r_2 = rsize_h + 1.0
            #mask_dens_h =  ((Dist_cylind  > r_1) &(Dist_cylind  < r_2)& (np.sqrt(rz**2) <= rsize_h))
            mask_dens_h =  ((np.sqrt(Dist_sphere)  > r_1) &(np.sqrt(Dist_sphere) < r_2))
            #mass_dens = np.sum(mgas[mask_dens_h])
            #densr = (mass_dens / (Pi*rsize_h*(r_2**2 - r_1**2))) * M_sun/(kpc_cm**3) # gr cm-3 or cm-2 surface density
            densr = np.mean(densgas[mask_dens_h]) #ICM is just hot gas

            mask2_shp = ( (np.sqrt(Dist_sphere)<= rsize_h)) # ICM include in estimate total halo mass M200
            mask2_dm_shp = ( (np.sqrt(Dist_sphere_dm)<= rsize_h))
            mask2_s_sph = ( (np.sqrt(Dist_sphere_s)<= rsize_h))

            mask2 =((np.sqrt(Dist_sphere)<= rsize_h))# ICM excluded
            mask2_dm = ( (np.sqrt(Dist_sphere_dm)<= rsize_h)& (np.sqrt((vx_dm)**2 + (vy_dm)**2 + (vz_dm)**2)<= Vesc))
            mask2_s = ( (np.sqrt(Dist_sphere_s)<= rsize_h)& (np.sqrt((vx_s)**2 + (vy_s)**2 + (vz_s)**2)<= Vesc))

            massstar_sph = np.sum(mstar[mask2_s_sph])
            massdm_sph = np.sum(mdm[mask2_dm_shp])
            massgas = np.sum(mgas[mask2])
            massdm = np.sum(mdm[mask2_dm])
            massstar = np.sum(mstar[mask2_s])
            massgas_sph = np.sum(mgas[mask2_shp])
            massr = massstar +  massdm + massgas
            M200 = np.log10(massstar_sph +  massdm_sph + massgas_sph)

            frestor_vir = (2.*Grav) * (massr) * densr / (rsize_h)  #[g s-2 cm-1]
            frestor_vir_dm = (2.*Grav) * (massdm_sph) * densr / (rsize_h)
            frestor_vir_gas = (2.*Grav) * (massgas) * densr / (rsize_h)
            frestor_vir_star = (2.*Grav) * (massstar_sph) * densr / (rsize_h)
            #print '*******************',densr, mass_dens,np.log10(frestor_vir*1e10), np.log10(ram*1e10)
            # Re --> total_half_mass restoring force
            rsize = radii



            r_1 = rsize - 1.0
            r_2 = rsize + 1.0
            #mask_dens_re =  ((Dist_cylind  > r_1) &(Dist_cylind  < r_2)& (np.sqrt(rz**2) <= rsize_h))
            #mass_dens = np.sum(mgas[mask_dens_re])
            #densr_re = (mass_dens / (Pi*rsize*(r_2**2 - r_1**2))) * M_sun/(kpc_cm**3)

            #mask_dens_re = ((Dist_cylind  < (rsize)**2) &(np.sqrt((vx_g)**2 + (vy_g)**2 + (vz_g)**2)<= Vesc) & (rz**2  < (rsize)**2))
            #mask_dens_re =  ((Dist_cylind  > r_1)&(Dist_cylind  < r_2))
            mask_dens_re =  ((np.sqrt(Dist_sphere)  > r_1) &(np.sqrt(Dist_sphere) < r_2))
            densr_re = np.mean(densgas[mask_dens_re]) #ICM is just hot gas

            #mask2_re = ((Dist_cylind  <= rsize)& (np.sqrt(rz**2) <= rsize))
            mask2_re =((np.sqrt(Dist_sphere)<= rsize))
            mask2_dm_re = ((np.sqrt(Dist_sphere_dm)<= rsize))
            mask2_s_re = ((np.sqrt(Dist_sphere_s)<= rsize))


            #rsize_p = (rsize* np.cos(teta1 - teta2))
            massstar_re = np.sum(mstar[mask2_s_re])
            massdm_re = np.sum(mdm[mask2_dm_re])
            massgas_re = np.sum(mgas[mask2_re])
            massr_re = massstar_re +  massdm_re + massgas_re
            frestor_re = (2.*Grav) * (massr_re) * densr_re / (rsize)  #1e10[g s-2 cm-1]
            frestor_re_dm = (2.*Grav) * (massdm_re) * densr_re / (rsize)
            frestor_re_gas = (2.*Grav) * (massgas_re) * densr_re / (rsize)
            frestor_re_star = (2.*Grav) * (massstar_re) * densr_re / (rsize)
            #print '*******************',densr_re, mass_dens,np.log10(frestor_re*1e10), np.log10(ram*1e10)
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

 #save the first min frestor_trunc
            bin_radii = -1.0
            # rtrunc_all = [];gap = [];ramii = [];gid = []
            for i in np.arange(2.0*(radii_h),1.,bin_radii):

              if i < rsize_h:
                maski = ( (np.sqrt(Dist_sphere)<= i))
                maski_dm = ( (np.sqrt(Dist_sphere_dm) <= i)& (np.sqrt((vx_dm)**2 + (vy_dm)**2 + (vz_dm)**2)<= Vesc))
                maski_s = ( (np.sqrt(Dist_sphere_s) <= i)& (np.sqrt((vx_s)**2 + (vy_s)**2 + (vz_s)**2)<= Vesc))
              else:
                maski = ( (np.sqrt(Dist_sphere)<= rsize_h))
                maski_dm = ( (np.sqrt(Dist_sphere_dm) <= rsize_h)& (np.sqrt((vx_dm)**2 + (vy_dm)**2 + (vz_dm)**2)<= Vesc))
                maski_s = ( (np.sqrt(Dist_sphere_s) <= rsize_h)& (np.sqrt((vx_s)**2 + (vy_s)**2 + (vz_s)**2)<= Vesc))

              #mask_densi =((Dist_cylind  > ((i)-1)) &(Dist_cylind  < ((i)+1))& (np.sqrt(rz**2) <= rsize_h))

              #densi = np.mean(densgas[mask_densi])
              r_1 = i - 1
              r_2 = i + 1
              #mask_densi =  ((Dist_cylind  > r_1) &(Dist_cylind  < r_2)& (np.sqrt(rz**2) <= rsize_h))

              #mass_dens = np.sum(mgas[mask_densi])
              #densi = (mass_dens / (Pi*rsize_h*(r_2**2 - r_1**2))) * M_sun/(kpc_cm**3)# gr cm-3 or cm-2 surface density
              mask_densi =  ((np.sqrt(Dist_sphere)  > r_1) &(np.sqrt(Dist_sphere) < r_2))
              densi = np.mean(densgas[mask_densi])

              massdmi = np.sum(mdm[maski_dm])
              massstari = np.sum(mstar[maski_s])
              massgasi = np.sum(mgas[maski])

              massi = massdmi + massstari + massgasi
              rami = (2.*Grav) * (massi) * densi / (i)

              #if (np.log10(ram*1e10) < np.log10(rami*1e10)): CHANGE RP < RF
              if (np.log10(ram*1e10) < np.log10(rami*1e10)):
                    rtrunc = (i)
                    gap = ((np.log10(rami*1e10) - np.log10(ram*1e10)))
                    ramii = (rami)
                    print '**********Rtrunc',rtrunc/rsize_h,'Ram',np.log10(ram*1e10),'gap', gap, 'Frest',np.log10(rami*1e10)
                    break
              else:
                    rtrunc = (0.0)
                    gap = ((np.log10(rami*1e10) - np.log10(ram*1e10)))
                    ramii = (rami)
            print 'Rtrunc',rtrunc/rsize_h,'Ram',np.log10(ram*1e10),'gap', gap, 'Frest',np.log10(rami*1e10)
            # zzz = np.array([rtrunc_all,gap,np.log10(ramii)+10]).T

            # if len(zzz[:,1]) > 0:
             # w2 = np.where((zzz[:,1] == np.nanmin(zzz[:,1])))[0]
             # if len(w2)>0:
                # rtrunc =   zzz[w2[0],0]
                # frestor_trunc = zzz[w2[0],2]
                # gap1 = zzz[w2[0],1]
             # else:
                # rtrunc = 0
                # frestor_trunc = 0

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
            #save as array in file
#---------------------------------------------------------------------------------------

            # bin_radii = -1.0
            # rtrunc_all = [];gap = [];ramii = [];gid = [];mh200 = [];rsize1 = [];cent_1 = [];gap_dm = [];gap_g = [];ramii_h = [];ramii_g = [];densi_all= []; massdmi_all= []; massgasi_all= []; massstari_all= [];ramii_s= [];rsize_vir=[]
            #
            # for i in np.arange(2.0*(radii_h),1.,bin_radii):
            #
            #   if i < rsize_h:
            #     maski = ( (np.sqrt(Dist_sphere)<= i))
            #     maski_dm = ( (np.sqrt(Dist_sphere_dm) <= i)& (np.sqrt((vx_dm)**2 + (vy_dm)**2 + (vz_dm)**2)<= Vesc))
            #     maski_s = ( (np.sqrt(Dist_sphere_s) <= i)& (np.sqrt((vx_s)**2 + (vy_s)**2 + (vz_s)**2)<= Vesc))
            #   else:
            #     maski = ( (np.sqrt(Dist_sphere)<= rsize_h))
            #     maski_dm = ( (np.sqrt(Dist_sphere_dm) <= rsize_h)& (np.sqrt((vx_dm)**2 + (vy_dm)**2 + (vz_dm)**2)<= Vesc))
            #     maski_s = ( (np.sqrt(Dist_sphere_s) <= rsize_h)& (np.sqrt((vx_s)**2 + (vy_s)**2 + (vz_s)**2)<= Vesc))
            #
            #   #mask_densi =((Dist_cylind  > ((i)-1)) &(Dist_cylind  < ((i)+1))& (np.sqrt(rz**2) <= rsize_h))
            #   #densi = np.mean(densgas[mask_densi])
            #   r_1 = i - 1
            #   r_2 = i + 1
            #   #mask_densi =  ((Dist_cylind  > r_1) &(Dist_cylind  < r_2)& (np.sqrt(rz**2) <= rsize_h))
            #   #mass_dens = np.sum(mgas[mask_densi])
            #   #densi = (mass_dens / (Pi*rsize_h*(r_2**2 - r_1**2))) * M_sun/(kpc_cm**3)
            #   mask_densi =  ((np.sqrt(Dist_sphere)  > r_1) &(np.sqrt(Dist_sphere) < r_2))
            #   densi = np.mean(densgas[mask_densi])
            #
            #   massdmi = np.sum(mdm[maski_dm])
            #   massstari = np.sum(mstar[maski_s])
            #   massgasi = np.sum(mgas[maski])
            #
            #   massi = massdmi + massstari + massgasi
            #   rami = (2.*Grav) * (massi) * densi / (i)
            #   rami_h = (2.*Grav) * (massgasi) * densi / (i)
            #   rami_g = (2.*Grav) * (massdmi) * densi / (i)
            #   rami_s = (2.*Grav) * (massstari) * densi / (i)
            #
            #   rtrunc_all.append(i)
            #   gap.append((np.log10(rami*1e10) - np.log10(ram*1e10)))
            #   ramii.append(rami)
            #   gid.append(galID[ii])
            #   mh200.append(mh[ii])
            #   rsize1.append(rsize)
            #   rsize_vir.append(rsize_h)
            #   cent_1.append(galcen[ii])
            #   gap_dm.append((np.log10(rami_h*1e10) - np.log10(rami*1e10)))
            #   gap_g.append((np.log10(rami_g*1e10) - np.log10(rami*1e10)))
            #   ramii_h.append(rami_h)
            #   ramii_g.append(rami_g)
            #   ramii_s.append(rami_s)
            #   densi_all.append(np.log10(densi))
            #   massdmi_all.append(np.log10(massdmi))
            #   massgasi_all.append(np.log10(massgasi))
            #   massstari_all.append(np.log10(massstari))
            #
            #
            #
            # zz = np.array([gid,rtrunc_all,gap,np.log10(ramii)+10,rsize1,rsize_vir,np.log10(mh200),cent_1,gap_dm,gap_g,np.log10(ramii_h)+10,np.log10(ramii_g)+10,np.log10(ramii_s)+10,densi_all, massdmi_all, massgasi_all, massstari_all]).T
            #
            # np.savetxt(f, zz, fmt='%i %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f',delimiter='\t')
            #f.close()
#---------------------------------------------------------------------------------------
            #write the particels data for individual halo
#---------------------------------------------------------------------------------------

            #f2 = open('./result/SIMBA_3004_get_gal/coordinate_ID2.asc', 'w+')
            #f2.write('# x y z vx vy vz'+"\n")
            #f3 = open('./result/SIMBA_3004_get_gal/coordinate_density_ID2.asc', 'w+')
            #f3.write('# x y z'+"\n")

            #mask_x = ( (np.sqrt(Dist_sphere)<= rsize_h))
            #coord = np.array([rx[mask_x], ry[mask_x],rz[mask_x],vx_g[mask_x],vy_g[mask_x],vz_g[mask_x]]).T
            #np.savetxt(f2, coord, fmt='%1.8f %1.8f %1.8f %1.8f %1.8f %1.8f',delimiter='\t')
            ##f2.close()

            ##maski_x = ((Dist_cylind  < (rsize_h* np.cos(teta1 - teta2))**2) &(rz**2  < (0.5*rsize_h**2)) &(Dist_cylind+ rz**2  > 0.0) )

            ##maski_x = ((Dist_cylind  < ((rsize_h)* np.cos(teta1 - teta2)+5)**2) &(Dist_cylind  > ((rsize_h)* np.cos(teta1 - teta2)-5)**2) &(np.sqrt((vx_g)**2 + (vy_g)**2 + (vz_g)**2)<= Vesc) & (rz**2  < (rsize_h)**2))


            ##maski_x =  (( Dist_project < rsize_h*np.sin(phi)))
            #r_1 = rsize_h-10.0
            #r_2 = rsize_h
            #maski_x =  ((Dist_cylind  > r_1) &(Dist_cylind  < r_2)& (np.sqrt(rz**2) <= rsize_h))

            ##maski_x = ((Dist_cylind  < (rsize_h)**2) & (Dist_sphere  < (rsize_h)**2))
            #coord3 = np.array([rx[maski_x], ry[maski_x],rz[maski_x]]).T
            #np.savetxt(f3, coord3, fmt='%1.8f %1.8f %1.8f',delimiter='\t')
            ##f3.close()
            #f, ((ax1)) = plt.subplots(1, 1,figsize=(6,5))
            #plt.scatter(rtrunc_all,np.log10(ramii)+10,color='red', s=3, alpha=0.99,label='_nolegend_')
            #yy = [np.log10(ram*1e10),np.log10(ram*1e10),np.log10(ram*1e10)]
            #xx = [-10 , 10 , 400]
            #plt.plot(xx,yy,'k--')
            #yy = [np.log10(frestor_vir*1e10),np.log10(frestor_vir*1e10),np.log10(frestor_vir*1e10)]
            #plt.plot(xx,yy,'k--')
            #f2.close()
            #f3.close()
            #xx = [rsize_vir,rsize_vir,rsize_vir]
            #yy = [-14,-13,-11]
            #plt.plot(xx,yy,'k--')

            #plots = './result/SIMBA_3004_get_gal/'
            #filename = plots+ 'frest_profile_test'
            #plt.savefig(filename + '.pdf')
            #plt.close()


            #ax = plt.axes(projection='3d')
            #ax.scatter(rx[mask_x], ry[mask_x],rz[mask_x], c = 'blue',s = 1,alpha = 0.05, linewidth=0.5)
            #ax.scatter(rx[maski_x], ry[maski_x],rz[maski_x], c='red',s = 1,alpha = 0.5, linewidth=0.5);


            #filename = plots+ '3D_gas_ID2'
            #plt.savefig(filename + '.pdf')
            #plt.close()
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

            #outflow & inflow gas---------------------------------------------------------------------------------------
            gasVelx = np.sum(mgas_all[mask2]*velgas_all[:,0][mask2])/np.sum(mgas_all[mask2])
            gasVely = np.sum(mgas_all[mask2]*velgas_all[:,1][mask2])/np.sum(mgas_all[mask2])
            gasVelz = np.sum(mgas_all[mask2]*velgas_all[:,2][mask2])/np.sum(mgas_all[mask2])
            vx_gas = velgas_all[:,0][mask2] - gasVelx
            vy_gas = velgas_all[:,1][mask2] - gasVely
            vz_gas = velgas_all[:,2][mask2] - gasVelz
            outf = np.where((vx_gas*rx[mask2] +vy_gas*ry[mask2] + vz_gas*rz[mask2]) > 0)[0]
            inf = np.where((vx_gas*rx[mask2] +vy_gas*ry[mask2] + vz_gas*rz[mask2]) < 0)[0]
            ngas = len(mgas_all[mask2])
        else:
            ram = 1e-19

        if ((len(mgas_c) > 0)):
            galPosx = g.pos.d[0]
            galPosy = g.pos.d[1]
            galPosz =  g.pos.d[2]
            vx = g.vel.d[0]
            vy = g.vel.d[1]
            vz = g.vel.d[2]

            # cold Gas accretion for Rvir<r< 2Rvir
            mask_c = ( (np.sqrt( (posgas_c[:,0] - galPosx)**2 + (posgas_c[:,1] - galPosy)**2 + (posgas_c[:,2] - galPosz)**2)> radii) &(np.sqrt( (posgas_c[:,0] - galPosx)**2 + (posgas_c[:,1] - galPosy)**2 + (posgas_c[:,2] - galPosz)**2) <= 2.0*radii))

            gasVelx_c = np.sum(mgas_c[mask_c]*velgas_c[:,0][mask_c])/np.sum(mgas_c[mask_c])
            gasVely_c = np.sum(mgas_c[mask_c]*velgas_c[:,1][mask_c])/np.sum(mgas_c[mask_c])
            gasVelz_c = np.sum(mgas_c[mask_c]*velgas_c[:,2][mask_c])/np.sum(mgas_c[mask_c])

            vx_gas = velgas_c[:,0][mask_c] - gasVelx_c
            vy_gas = velgas_c[:,1][mask_c] - gasVely_c
            vz_gas = velgas_c[:,2][mask_c] - gasVelz_c
            rx = posgas_c[:,0][mask_c] - galPosx
            ry = posgas_c[:,1][mask_c] - galPosy
            rz = posgas_c[:,2][mask_c] - galPosz
            Macc_c = np.sum((mgas_c[mask_c]/radii) * (vx_gas*rx +vy_gas*ry + vz_gas*rz)/np.sqrt((rx)**2 + (ry)**2 + (rz)**2))

            # cold gas ram pressure
            dens_c = np.sum(mgas_c[mask_c]*densgas_c[mask_c])/np.sum(mgas_c[mask_c])
            #dens_c = np.mean(densgas)
            vel_wrt_gas2_c = (vx - gasVelx_c)**2 + (vy - gasVely_c)**2 + (vz - gasVelz_c)**2
            ram_c = (dens_c * vel_wrt_gas2_c)
        else:
            ram_c = 0.0
            dens_c = 0.0
            vel_wrt_gas2_c = 0.0
            Macc_c = -999.9
        if ((len(mgas_h) > 0)):
            galPosx = g.pos.d[0]
            galPosy = g.pos.d[1]
            galPosz =  g.pos.d[2]
            vx = g.vel.d[0]
            vy = g.vel.d[1]
            vz = g.vel.d[2]

            # Hot Gas accretion for Rvir<r< 2Rvir
            mask_h = ( (np.sqrt( (posgas_h[:,0] - galPosx)**2 + (posgas_h[:,1] - galPosy)**2 + (posgas_h[:,2] - galPosz)**2)> radii) &(np.sqrt( (posgas_h[:,0] - galPosx)**2 + (posgas_h[:,1] - galPosy)**2 + (posgas_h[:,2] - galPosz)**2) <= 2.0*radii))

            gasVelx_h = np.sum(mgas_h[mask_h]*velgas_h[:,0][mask_h])/np.sum(mgas_h[mask_h])
            gasVely_h = np.sum(mgas_h[mask_h]*velgas_h[:,1][mask_h])/np.sum(mgas_h[mask_h])
            gasVelz_h = np.sum(mgas_h[mask_h]*velgas_h[:,2][mask_h])/np.sum(mgas_h[mask_h])

            vx_gas = velgas_h[:,0][mask_h] - gasVelx_h
            vy_gas = velgas_h[:,1][mask_h] - gasVely_h
            vz_gas = velgas_h[:,2][mask_h] - gasVelz_h
            rx = posgas_h[:,0][mask_h] - galPosx
            ry = posgas_h[:,1][mask_h] - galPosy
            rz = posgas_h[:,2][mask_h] - galPosz
            Macc_h = np.sum((mgas_h[mask_h]/radii) * (vx_gas*rx +vy_gas*ry + vz_gas*rz)/np.sqrt((rx)**2 + (ry)**2 + (rz)**2))
            # Hot gas ram pressure
            dens_h = np.sum(mgas_h[mask_h]*densgas_h[mask_h])/np.sum(mgas_h[mask_h])
            #dens_h =np.mean(densgas[mask_h])
            vel_wrt_gas2_h = (vx - gasVelx_h)**2 + (vy - gasVely_h)**2 + (vz - gasVelz_h)**2
            ram_h = (dens_h * vel_wrt_gas2_h)
        else:
            ram_h = 0.0
            dens_h = 0.0
            vel_wrt_gas2_h = 0.0
            Macc_h = -999.9

        # appending ---------------------------------------------------------------------------------------

        dens_r.append(np.log10(densr))
        mass_r.append(massr)
        mass_r_re.append(massr_re)
        dens_r_re.append(np.log10(densr_re))

        r_size.append(rsize)
        vel_wrt2.append(vel_wrt_gas2)
        dens_g.append(np.log10(dens))
        ram_p.append(np.log10(ram*1e10)) # in g/cm/s2
        r_trunc.append(rtrunc)

        f_restor_vir.append(np.log10(frestor_vir*1e10))
        f_restor_vir_dm.append(np.log10(frestor_vir_dm*1e10))
        f_restor_vir_gas.append(np.log10(frestor_vir_gas*1e10))
        f_restor_vir_star.append(np.log10(frestor_vir_star*1e10))
        f_restor_re.append(np.log10(frestor_re*1e10))
        f_restor_re_dm.append(np.log10(frestor_re_dm*1e10))
        f_restor_re_gas.append(np.log10(frestor_re_gas*1e10))
        f_restor_re_star.append(np.log10(frestor_re_star*1e10))
        # f_restor_trunc.append(frestor_trunc)

        vel_wrt2_c.append(vel_wrt_gas2_c)
        dens_g_c.append(np.log10(dens_c))
        ram_p_c.append(np.log10(ram_c*1e10)) # in g/cm/s2

        vel_wrt2_h.append(vel_wrt_gas2_h)
        dens_g_h.append(np.log10(dens_h))
        ram_p_h.append(np.log10(ram_h*1e10)) # in g/cm/s2


        M_acc_cold.append(Macc_c*1e-9) # M_sun/year
        M_acc_hot.append(Macc_h*1e-9) # M_sun/year
         #outflow andf inflowe of gas in the virial radius

        outflow.append(len(outf))
        inflow.append(len(inf))
        all_gas.append(len(outf)+len(inf))
        # SF, Wind and Jet gas particles
        nsf.append(len(sfgnh))
        nwd.append(len(wdgnh))
        njet.append(len(jetgnh))

        hc_dist_center.append(gal_dist_center)
        hc_r200.append(hcr200)
        r_200.append(radii_h)
        M_200.append(M200)
        redshift1.append(redshift)
        snap1.append(snap)
        Temp.append(np.log10(np.median(gasTemp)))
        # print '------- ramp, gascount,Macc_c,Macc_h:',ram*1e10,ngas, Macc_c*1e-9, Macc_h*1e-9
        print galID[ii],'--, ramp1, rtrunc, rsize,rvir, cent,Fr_vir,Fr_re',np.log10(ram*1e10), rtrunc,rsize,radii_h, galcen[ii],np.log10(frestor_vir*1e10),np.log10(frestor_re*1e10)

        ii += 1
        #break
    # with halo props ---------------------------------------------------------------------------------------
    # f.close()
    cc = np.array([galID,snap1,redshift1,galPos[:,0],galPos[:,1],galPos[:,2],np.log10(galMstar),np.log10(mh),galsfr,galHImass,
    galH2mass,galcen,ram_p,vel_wrt2,dens_g,nsf,nwd,njet,outflow,inflow,all_gas,Temp,M_acc_cold,M_acc_hot,ram_p_c,ram_p_h,vel_wrt2_c,vel_wrt2_h,dens_g_c,dens_g_h,dens_r,mass_r,r_size,f_restor_vir,r_trunc,f_restor_vir_gas,f_restor_vir_dm,hc_dist_center,hc_r200,f_restor_vir_star,f_restor_re,f_restor_re_dm,f_restor_re_gas,f_restor_re_star,mass_r_re,dens_r_re,r_200,M_200]).T
    np.savetxt(output, cc, fmt='%i %i %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %i %1.19f %1.19f %1.29f %i %i %i %i %i %i %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.32f %1.32f %1.32f %1.8f %1.8f %1.9f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f %1.8f',delimiter='\t',header='galID\tsnap\tredshift\tx\ty\tz\tlogMstar\tlogMhalo\tsfr\tMassHI\tMassH2\tcentral\tram_p\tvelwrt2\tdensg\tNSF\tNwind\tNjet\toutflowvir\tinflowvir\tallgasvir\tTemp\tM_acc_cold\tM_acc_hot\tram_p_c\tram_p_h\tvel_wrt2_c\tvel_wrt2_h\tdens_g_c\tdens_g_h\tdens_r\tmass_r\tr_size\tf_restor_vir\tr_trunc\tf_restor_vir_gas\tf_restor_vir_dm\thc_dist_center\thc_r200\tf_restor_vir_star\tf_restor_re\tf_restor_re_dm\tf_restor_re_gas\tf_restor_re_star\tmass_r_re\tdens_r_re\tR200\tM200')
