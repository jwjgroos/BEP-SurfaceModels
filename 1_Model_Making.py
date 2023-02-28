# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:16:39 2021

@author: Jim Groos 
"""

#Import libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as mat_io
import matplotlib.colors as clr
from matplotlib.path import Path
from scipy import signal
from scipy import ndimage

plt.close('all')

# data dir
data_dir = './data/'

# Load the colormap and convert to numpy arrays
temp = mat_io.loadmat(data_dir + 'cgray.mat')
cgray = temp['cgray']
temp = mat_io.loadmat(data_dir + 'cmap.mat')
cmap = temp['cmap']
temp = None

# Define the model parameters
nx = 128 # 128 
dx = 20
nz = 64 #64
dz = 10



#### Important arguments ####

# If batch = True: use powershell
# If batch = False: use spyder 
batch = True

# Number of images to be created (max = 90) 
if batch ==  False:
    
    # spyder: specify number of first image and number of last image
    first_image = 1
    last_image  = 1
    
elif len(sys.argv) >= 3:
    
    # powershell: see README.txt
    first_image = int(sys.argv[1])
    last_image  = int(sys.argv[2])
    
    if (len(sys.argv)) >= 4:
        np.random.seed(int(sys.argv[3]))
        
else:
    assert False, 'Not enough arguments <first image> <last_image> <random seed (optional)>'
    
########Choices to make within the model########

# Define the random amount of layers in a model
n_layers_min = 8
n_layers_max = 10

#Define the range of the density as a percentage of the velocity
density_low = 60  # in %
density_high = 80 # in %

# Define the depth level of the first layer, the water layer, as a percentage of the total
perc_waterLayerLow = 10  # in %
perc_waterLayerHigh = 40 # in %

# Define the amount of pertubations per layer
n_minPertubations = 9
n_maxPertubations = 10  

# Change the std of the pertubations: Higer results in more rejected layers, lower means less random
normalRangePerLayer = 20

# Polyfit order for the boundary line (default = 5)
order = 5

## Salt structure ##
# velocity inside the salt
salt_structure = True
v_salt_low = 3300
v_salt_high = 3700

# Random number of corners the salt structure could have (default = 3,6)
n_points_low = 3 
n_points_high = 6

# Save gradient velocity
v_con_low = 1500
v_con_high = 2100





## Start Code ##
## No parameters ## 
# Initialize some important values
reject_image= False

n_images = last_image
n_image = first_image-1


# Start with making images
while n_image < n_images:
    print('Current image = ' + str(n_image+1))
    
    #define the random amount of layers of a model 
    n_layers = np.random.randint(n_layers_min,n_layers_max, size=1)
    
    # define the background velocities and density
    # The velocity is randomly chosen. First layer is always 1500 m/s, rest is increasing with every layer. See BEP for table of current settings.
    v = np.arange(1750,n_layers*250+1750,step=250)
    v += np.random.randint(-150,150,len(v))
    rho = v*np.random.uniform(density_low/100,density_high/100,size=(len(v)))
    v = np.insert(v,0,1500)
    rho = rho.astype(int) 
    rho = np.insert(rho,0,1000)
    
    #create randomized depth levels
    # water layer
    layer_1 = np.random.uniform(perc_waterLayerLow,perc_waterLayerHigh)/100
    
    # rest of the layers, uniform randomnly spread over the rest of the model. 
    layer_2n = np.random.uniform(0.1,10,int(n_layers-1))
    layer_2n = np.cumsum((layer_2n/sum(layer_2n)))*(1-layer_1)+(layer_1)
    
    # create depth array
    z=np.append(layer_1,layer_2n)
    z= np.round(z*nz)
    z= np.insert(z,0,0)
    z= z.astype(int)
    
    # Initialize vel den and imag arrays
    vel = np.zeros((nz, nx))
    imag = np.zeros((nz+1, nx))
    den = np.zeros((nz,nx))
    
    # fill background velocities
    for iz in range(z.size-1):
        vel[z[iz]:z[iz+1], :] = v[0]
        den[z[iz]:z[iz+1], :] = rho[0]
    
    # add pertubations: first determine the amount of pertubations per layer (this to create curved boundaries)
    n_PertubationsPerLayer = np.random.randint(n_minPertubations,n_maxPertubations,size=int(n_layers-1))
    
    # Assign x_values and z_values where the random pertubations happen
    # Initialize arrays
    x_pertubations = np.zeros(shape=(int(n_layers-1),np.max(n_PertubationsPerLayer)))
    z_pertubations = np.copy(x_pertubations)
    
    # For every layer random points are assigned for the curved boundary
    for ix in range(int(n_layers-1)):
        x_pertubations[ix][0:n_PertubationsPerLayer[ix]] = np.random.randint(1,nx-1,n_PertubationsPerLayer[ix])
        z_pertubations[ix][0:n_PertubationsPerLayer[ix]] = np.round(np.random.normal(loc=z[ix+1],scale=normalRangePerLayer,size=n_PertubationsPerLayer[ix]))
    x_pertubations = np.squeeze(x_pertubations)

    # Plot every layer onto the other layer, with curved boundaries
    z_boundaries1 = np.zeros(nx)
    for layer in range(0,int(n_layers-1)):
        
        ##  Create a line through these pertubations ##
        # Initizalize array where the boundaries can be stored in
        z_pertboundaries = np.zeros(nx)
        z_pertboundaries[:] = z[layer+1]
        
        # put in the extra values
        for i in range(x_pertubations.shape[1]):
            z_pertboundaries[int(x_pertubations[layer][i])] = z_pertubations[layer][i]
        
        # fix the first index, it tends to get stuck on zero
        z_pertboundaries[0] = z[layer+1]
        
        # Creating the curve through these pertubations
        x_array = np.linspace(0,dx*nx, num=nx)
        poly = np.polyfit(x_array,z_pertboundaries,order)
        poly1d = np.poly1d(poly)
        z_boundaries = np.round(poly1d(x_array))
        ##################################################
        
        # Reject the image if layers overlap, rejection will be handled later.
        test = (z_boundaries<=z_boundaries1)
        if np.sum(test) >= 1:
            reject_image = True
            reason = 'Overlapping Layers'
            break
        z_boundaries1= np.copy(z_boundaries)
        
        # Add the pertubations onto the velocity model
        pert = np.zeros((nz,nx))
        pertden = np.zeros((nz,nx))
        for k in range(nz):
            for l in range(nx):
                if z_boundaries[l] <= k:
                    pert[k][l] = v[layer+1]
                    pertden[k][l] = rho[layer+1]
        
        # Create the density and velocity arrays            
        for k in range(pert.shape[0]):
            for l in range(pert.shape[1]):
                if pert[k][l] != 0:
                    vel[k][l] = pert[k][l]
                    den[k][l] = pertden[k][l]
                    
    ## Adding Salt structure ##
    # random points are chosen, never in water layer.
    # then using a centerpoint lines are drawn from every point in such a way that there are no overlapping lines. 
    if salt_structure == True: 
        
        # Add random salt structure onto the model
        n = np.random.randint(n_points_low,n_points_high)
        
        x = np.random.randint(0,nx,n)
        y = np.random.randint(layer_1*nz,nz,n)
        
        ##computing the (or a) 'center point' of the polygon
        center_point = [np.sum(x)/n, np.sum(y)/n]
        
        angles = np.arctan2(x-center_point[0],y-center_point[1])
        
        ##sorting the points:
        sort_tups = sorted([(i,j,k) for i,j,k in zip(x,y,angles)], key = lambda t: t[2])
        
        ##making sure that there are no duplicates:
        if len(sort_tups) != len(set(sort_tups)):
            raise Exception('two equal coordinates -- exiting')
        
        x,y,angles = zip(*sort_tups)
        x = list(x)
        y = list(y)
        
        ##appending first coordinate values to lists:
        x.append(x[0])
        y.append(y[0])
        
        codes = [
        Path.MOVETO
        ]
        for i in range(len(x)-2):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        
        tuples = []
        for i in range(len(x)):
            tuples.append((x[i],y[i]))
        
        x, y = np.meshgrid(np.arange(nx), np.arange(nz)) # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T 
        
        p = Path(tuples,codes) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(nz,nx)
        
        v_salt = np.random.randint(v_salt_low,v_salt_high)
        rho_salt = v_salt*np.random.uniform(density_low/100,density_high/100)
        salt_pertubation_vel = mask*v_salt
        salt_pertubation_den = mask*rho_salt
        
        for k in range(salt_pertubation_vel.shape[0]):
            for l in range(salt_pertubation_vel.shape[1]):
                if salt_pertubation_vel[k][l] != 0:
                    if k < layer_1*nz:
                        reject_image = True
                        reason = 'Salt Pertubation Overlaps'
                    else: 
                        vel[k][l] = salt_pertubation_vel[k][l]
                        den[k][l] = salt_pertubation_den[k][l]
                
    # Check if an image should be rejected. If so, break iteration and try again
    if reject_image == True:
        reject_image = False
        print('image ' + str(n_image+1) + ' will be rejected: ' + reason)
        continue
    
    # If not
    n_image += 1
    
    # create the reflectivity image
    imp = den*vel
    for iz in range(1,nz):
        imag[iz,:] = (imp[iz,:] - imp[iz-1,:]) / (imp[iz,:] + imp[iz-1,:])
        
    # plot 1D plots of vel and image
    i_loc = 51
    x_ax = np.arange(0, nx*dx, dx)
    z_ax = np.arange(0, nz*dz, dz)
    
    n_gaus = 1
    vel0 = 1/ndimage.gaussian_filter(1/vel,n_gaus,order=0)
    
    # create gradient velocity model
    temp = np.linspace(v_con_low, v_con_high, nz)
    velgrad = np.zeros((nz,nx))
    velgrad = velgrad + temp[:,None]
        

    # plot true velocity 
    plt.figure()
    plt.imshow(vel, cmap=clr.ListedColormap(cmap), vmin=np.amin(vel)-100, vmax=v_salt_high+100, extent=(0,2000,500,0), aspect='auto')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.colorbar()
    plt.title('True velocity')
    plt.tight_layout()
    plt.savefig('./data/figures/Module_A/True_Velocity/True_Velocity_'+ str(n_image)+'.png')
    plt.close()
    
    plt.figure()
    plt.imshow(den, cmap=clr.ListedColormap(cmap), vmin=np.amin(den)-100, vmax=np.amax(den)+100, extent=(0,2000,500,0), aspect='auto')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.colorbar()
    plt.title('True Density')
    plt.tight_layout()
    plt.savefig('./data/figures/Module_A/True_Density/True_Density_'+ str(n_image)+'.png')
    plt.close()
    
    # plot initial velocity
    plt.figure()
    plt.imshow(vel0, cmap='seismic', vmin=np.amin(vel)-100, vmax=v_salt_high+100, extent=(0,2000,500,0))
    plt.imshow(vel0, cmap=clr.ListedColormap(cmap), vmin=np.amin(vel)-100, vmax=np.amax(vel)+100,  extent=(0,2000,500,0), aspect='auto')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.colorbar()
    plt.title('Smooth: Gauss sigma = 1 ')
    plt.tight_layout()
    plt.savefig('./data/figures/Module_A/Vel0/Vel0_'+ str(n_image)+'.png')
    plt.close()
    
    # plot gradient velocity
    plt.figure()
    plt.imshow(velgrad, cmap='seismic', vmin=np.amin(vel)-100, vmax=v_salt_high+100, extent=(0,nx*dx,nz*dz,0))
    plt.imshow(velgrad, cmap=clr.ListedColormap(cmap), vmin=np.amin(vel)-100, vmax=np.amax(vel)+100,  extent=(0,nx*dx,nz*dz,0), aspect='auto')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.colorbar()
    plt.title('Smooth: Gauss sigma = 1 ')
    plt.tight_layout()
    plt.savefig('./data/figures/Module_A/Velgrad/Velgrad_'+ str(n_image)+'.png')
    plt.close()
    
    # plot reflectivity image
    plt.figure()
    plt.imshow(imag, cmap=clr.ListedColormap(cgray), extent=(0,2000,500,0), aspect='auto')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.colorbar()
    plt.title('True reflectivity image')
    plt.tight_layout()
    plt.savefig('./data/figures/Module_A/True_Reflectivity/True_Reflectivity_'+ str(n_image)+'.png')
    plt.close()


    vel_file = './data/vel/vel_' + str(n_image)
    vel0_file = './data/vel0/vel0_' + str(n_image)
    imag_file = './data/imag/imag_' + str(n_image)
    velgrad_file = './data/velgrad/velgrad_' + str(n_image)

    np.save(vel_file, vel)
    np.save(vel0_file, vel0)
    np.save(imag_file, imag)
    np.save(velgrad_file,velgrad)