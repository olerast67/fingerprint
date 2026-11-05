import cv2, numpy as np, sys, glob
res=[]
for f in sorted(glob.glob("samples/diversity/*.jpg")+glob.glob("samples/internet/*.jpg")):
    im=cv2.imread(f); H,W=im.shape[:2]
    ycc=cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
    skin=((ycc[...,1]>133)&(ycc[...,1]<180)&(ycc[...,2]>77)&(ycc[...,2]<135)).astype(np.float32)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    d=cv2.GaussianBlur(g,(0,0),1.5)-cv2.GaussianBlur(g,(0,0),4)
    gx=cv2.Sobel(cv2.GaussianBlur(g,(0,0),2),cv2.CV_32F,1,0);gy=cv2.Sobel(cv2.GaussianBlur(g,(0,0),2),cv2.CV_32F,0,1)
    T=300; best=None
    for y in range(0,H-T,T//2):
        for x in range(0,W-T,T//2):
            s=skin[y:y+T,x:x+T].mean()
            if s<0.9: continue
            e=d[y:y+T,x:x+T].std()
            jxx=(gx[y:y+T,x:x+T]**2).sum();jyy=(gy[y:y+T,x:x+T]**2).sum();jxy=(gx[y:y+T,x:x+T]*gy[y:y+T,x:x+T]).sum()
            coh=np.sqrt((jxx-jyy)**2+4*jxy**2)/(jxx+jyy+1e-6)
            sc=e*coh
            if best is None or sc>best[0]: best=(sc,x,y,e,coh)
    if best: res.append((best[0],f.split('/')[-1],best[1],best[2],round(best[3],2),round(best[4],2)))
for r in sorted(res,reverse=True)[:16]: print(r)
