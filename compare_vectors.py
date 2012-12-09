def comparevectors(test,control,toltest = 1e-20):
    rows,cols = test.shape
    crows,ccols = control.shape
    if rows != crows:
        return
    agree = True
    list = []
    for f in range(cols):
        if agree:
            for j in range(ccols):
                sd = 0
                xbar = 0
                for i in range(rows):
                    xbar += (1/rows)*test[i,f]/control[i,j]
                for i in range(rows):
                    sd += (1/rows)*(test[i,f]/control[i,j]-xbar)**2
                sd = sqrt(sd)
                if sd<tolres:
                    if j not in list:
                        list.append(j)
                        break
                    else:
                        continue
                    
            if sd>tolres:
                print "vectors disagree"
                agree = False
                break
    if agree: print "vectors agree"