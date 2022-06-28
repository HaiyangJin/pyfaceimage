
import pyfaceimage as fim

im1 = fim.image('test/multi/L1.png', read=True)
im2 = fim.image('test/multi/L2.png', read=True)

# default (ali, top)
cf_ali_top = fim.mkcf(im1, im2)
cf_ali_top.show()
cf_ali_top.save(extrafolder='tmp')

cf_ali_bot = fim.mkcf(im1, im2, cueistop=False)
cf_ali_bot.show()
cf_ali_bot.save(extrafolder='tmp')

cf_mis_top = fim.mkcf(im1, im2, misali=100)
cf_mis_top.show()
cf_mis_top.save(extrafolder='tmp')

cf_mis_bot = fim.mkcf(im1, im2, cueistop=False, misali=100)
cf_mis_bot.show()
cf_mis_bot.save(extrafolder='tmp')

# show cues
cf_ali_top_c = fim.mkcf(im1, im2, showcue=True)
cf_ali_top_c.show()
cf_ali_top_c.save(extrafolder='tmp')

cf_ali_bot_c = fim.mkcf(im1, im2, cueistop=False, showcue=True)
cf_ali_bot_c.show()
cf_ali_bot_c.save(extrafolder='tmp')

cf_mis_top_c = fim.mkcf(im1, im2, misali=100, showcue=True)
cf_mis_top_c.show()
cf_mis_top_c.save(extrafolder='tmp')

cf_mis_bot_c = fim.mkcf(im1, im2, cueistop=False, misali=100, showcue=True)
cf_mis_bot_c.show()
cf_mis_bot_c.save(extrafolder='tmp')

# test misaligned
cf_mis1 = fim.mkcf(im1, im2, misali=100)
cf_mis1.show()

cf_mis2 = fim.mkcf(im1, im2, misali=0.5)
cf_mis2.show()

cf_mis2 = fim.mkcf(im1, im2, misali=2.5)

# test make cf with imdict
imdict = fim.dir('test/multi')
cfdict = fim.dictim.mkcfs(imdict)
fim.checksample(cfdict).show()
fim.save(cfdict,extrafolder='tmp')
