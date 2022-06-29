import os
import pyfaceimage as fim

# import matplotlib.pyplot as plt

# tmp1 = fim.dir('test/intact', read=False)
# fim.dict.read(tmp1)
# fim.checksample(tmp1).show()

stimdict = fim.dir('test/multi')


bsdict = fim.deepcopy(stimdict)
fim.mkboxscr(bsdict)

fim.checksample(bsdict).show()
fim.save(bsdict, extrafolder='tmp')
[os.remove(f.outfile) for f in bsdict.values()]

fim.checksample(stimdict).show()

### Test
testdict = fim.deepcopy(stimdict)
fim.checksample(testdict).show()
fim.dictim.dictfunc('mkboxscr', testdict)