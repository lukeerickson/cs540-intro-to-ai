import sys
import math
import itertools


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    X=[0]*26
    print("Q1")
    with open (filename,encoding='utf-8') as f:
        for c in itertools.chain.from_iterable(f):
            # count instances of each letter
            if (c == 'a' or c == 'A'):
                X[0] += 1
            if (c == 'b' or c == 'B'):
                X[1] += 1
            if (c == 'c' or c == 'C'):
                X[2] += 1
            if (c == 'd' or c == 'D'):
                X[3] += 1    
            if (c == 'e' or c == 'E'):
                X[4] += 1
            if (c == 'f' or c == 'F'):
                X[5] += 1
            if (c == 'g' or c == 'G'):
                X[6] += 1
            if (c == 'h' or c == 'H'):
                X[7] += 1
            if (c == 'i' or c == 'I'):
                X[8] += 1
            if (c == 'j' or c == 'J'):
                X[9] += 1
            if (c == 'k' or c == 'K'):
                X[10] += 1
            if (c == 'l' or c == 'L'):
                X[11] += 1
            if (c == 'm' or c == 'M'):
                X[12] += 1
            if (c == 'n' or c == 'N'):
                X[13] += 1
            if (c == 'o' or c == 'O'):
                X[14] += 1
            if (c == 'p' or c == 'P'):
                X[15] += 1
            if (c == 'q' or c == 'Q'):
                X[16] += 1
            if (c == 'r' or c == 'R'):
                X[17] += 1
            if (c == 's' or c == 'S'):
                X[18] += 1
            if (c == 't' or c == 'T'):
                X[19] += 1
            if (c == 'u' or c == 'U'):
                X[20] += 1
            if (c == 'v' or c == 'V'):
                X[21] += 1
            if (c == 'w' or c == 'W'):
                X[22] += 1
            if (c == 'x' or c == 'X'):
                X[23] += 1
            if (c == 'y' or c == 'Y'):
                X[24] += 1
            if (c == 'z' or c == 'Z'):
                X[25] += 1    
                
        print("A " + str(X[0]))
        print("B " + str(X[1]))
        print("C " + str(X[2]))
        print("D " + str(X[3]))
        print("E " + str(X[4]))
        print("F " + str(X[5]))
        print("G " + str(X[6]))
        print("H " + str(X[7]))
        print("I " + str(X[8]))
        print("J " + str(X[9]))
        print("K " + str(X[10]))
        print("L " + str(X[11]))
        print("M " + str(X[12]))
        print("N " + str(X[13]))
        print("O " + str(X[14]))
        print("P " + str(X[15]))
        print("Q " + str(X[16]))
        print("R " + str(X[17]))
        print("S " + str(X[18]))
        print("T " + str(X[19]))
        print("U " + str(X[20]))
        print("V " + str(X[21]))
        print("W " + str(X[22]))
        print("X " + str(X[23]))
        print("Y " + str(X[24]))
        print("Z " + str(X[25]))

    return X


# how should comments look?
# think about printing one string
# do vimdiff in linux terminal
def main():
    X = shred("letter.txt")
    
    (e,s) = get_parameter_vectors()
    print("Q2")
    print(round(X[0]*math.log(e[0]),4))
    print(round(X[0]*math.log(s[0]),4))
    
    #F(English) = log.6 + 
    #F(Spanish) = log.4 + 
    
    # q3 is incorrect for some
    english = 0
    spanish = 0
    
    for x in range(0,25):
        english += X[x] * math.log(e[x])
        spanish += X[x] * math.log(s[x])
        
    english += math.log(.6)
    spanish += math.log(.4)

    print("Q3")
    print(round(english,4))  
    print(round(spanish,4))
    
    print("Q4")
    probEnglish = 0
    if(spanish - english >= 100):
        probEnglish = 0
    elif(spanish - english <= -100):
        probEnglish = 1
    else:
        probEnglish = 1 / (1 + math.exp(spanish - english))
    print(round(probEnglish,4))
    
    
main()
