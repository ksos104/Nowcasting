import numpy as np
from matplotlib import projections, pyplot as plt
from celluloid import Camera
import pdb
import numpy as np 
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import scipy.ndimage as ndimage

date = '201907260300'
vid = np.load('./save/' + date + '.npy')
vid = ndimage.rotate(vid, 180, reshape=True)

fig = plt.figure()
camera = Camera(fig)
ax = fig.add_subplot(projection = ccrs.PlateCarree())

for img in vid :
    ax.set_extent([122,131.1,43,33]) 
    ax.coastlines('50m')
    im = ax.imshow(img, extent=(122,131.1,43,33))
    ax.plot()
    ax.plot(extent=(122,131.1,43,33),interpolation='nearest',transform = ccrs.PlateCarree())
    camera.snap()


animation = camera.animate(interval=100, blit=True)
animation.save('sine.gif')