from utils import rsrp, Vector

def main():
    u = Vector(0,0,25)
    v = Vector(300,300,1.5)

    print(rsrp(u,v,30,35))

main()