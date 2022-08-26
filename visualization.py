import pdb
import numpy as np
import cartopy.crs as ccrs 
import scipy.ndimage as ndimage
import datetime as dt
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import projections, pyplot as plt
import cartopy.feature as cfeature

def ktpw_cmap():
    cols=[   [0.0,0.0,0.0],
             [0.91372549, 1.0, 0.89019608],
             [0.78823529, 0.99607843, 0.72941176],
             [0.61960784, 0.97647059, 0.52941176],
             [0.85490196, 0.91372549, 1.0],
             [0.78431373, 0.84313725, 0.98823529],
             [0.69803922, 0.78431373, 0.99607843],
             [0.56862745, 0.70588235, 0.95686275],
             [0.94509804, 0.8745098,  0.8745098],
             [0.88235294, 0.81176471, 0.81176471],
             [0.83137255, 0.69019608, 0.69019608],
             [0.74901961, 0.56078431, 0.55294118],
             [0.69019608, 0.43921569, 0.43137255],
             [0.55294118, 0.25490196, 0.25490196],
             [0.44313725, 0.0, 0.00392157],
             [0.26666667, 0.0, 0.0],
             [0.26666667, 0.0, 0.0],
             [0.0, 0.0, 0.0],
         ]
    
    lev = [0.0001, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0,70.0,80.0,90.0,100.0,150.0] 
    
    nil = cols.pop(0)
    under = cols[-1]
    over = cols.pop()
    
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil,alpha = 0.0)
    cmap.set_under(under,alpha = 0.0)
    cmap.set_over(over)
    
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)

    return cmap,norm


def sevir_cmap():
    """
    https://github.com/MIT-AI-Accelerator/sevir_challenges/blob/main/radar_nowcasting/RadarNowcastBenchmarks.ipynb
    """
    cols=[   [0,0,0],
             [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
             [0.1568627450980392,  0.7450980392156863,  0.1568627450980392],
             [0.09803921568627451, 0.5882352941176471,  0.09803921568627451],
             [0.0392156862745098,  0.4117647058823529,  0.0392156862745098],
             [0.0392156862745098,  0.29411764705882354, 0.0392156862745098],
             [0.9607843137254902,  0.9607843137254902,  0.0],
             [0.9294117647058824,  0.6745098039215687,  0.0],
             [0.9411764705882353,  0.43137254901960786, 0.0],
             [0.6274509803921569,  0.0, 0.0],
             [0.9058823529411765,  0.0, 1.0],
         ]
    lev = [16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    
    nil = cols.pop(0)
    under = cols[0]
    over = cols.pop()
    
    cmap=mpl.colors.ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    
    norm = mpl.colors.BoundaryNorm(lev, cmap.N)
    
    return cmap,norm

def visualize_clip(clip,save_path):
    date = Path(clip).stem
    clip = np.load(clip)
    cmap,norm = ktpw_cmap()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = ccrs.PlateCarree())
    ax.set_extent([119,134.9,29.5,45]) 
    ax.coastlines('50m')
    ax.add_feature(cfeature.LAND, facecolor='0.75')

    for i in range(len(clip)) :       
        d = dt.datetime.strptime(date, '%Y%m%d%H%M')
        im = ax.imshow(clip[i], extent=(116,136,29.5,46), cmap=cmap, norm = norm)
        ax.plot(interpolation='nearest',transform = ccrs.PlateCarree())
        ax.set_title(f"[GK2A TPW] {d+dt.timedelta(minutes=i*10)} UTC", 
                     backgroundcolor='black',size=11.2, color='white')

        cbar = plt.colorbar(im,fraction=0.046, pad=0.01)
        cbar.ax.set_title('  mm',fontsize=8)
        
        fig.savefig(save_path+f'frame_{i}.png')
        cbar.remove()


if __name__ == '__main__':
    clip = "/mnt/server11_hard4/jiny/Nowcasting/kTPW/train/202101150900.npy"
    Path('result').mkdir(parents=True, exist_ok=True)
    visualize_clip(clip, 'result/')