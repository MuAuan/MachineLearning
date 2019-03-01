from PIL import Image, ImageDraw

s=72

images = []
for i in range(0,360,5):
    im = Image.open('k-means/pca_example/tran_pca6d/pca3_PCA3d_angle_'+str(i)+'.jpg') 
    im =im.resize(size=(512, 512), resample=Image.NEAREST)  #- NEAREST - BOX - BILINEAR - HAMMING - BICUBIC - LANCZOS
    images.append(im)
    
images[0].save('./k-means/pca_example/tran_pca6d_pca3d512_72_n360.gif', save_all=True, append_images=images[1:s], duration=100*2, loop=0)    
