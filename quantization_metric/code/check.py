
sizes = [1725,1043,277,2224,1221,500]
# sizes = [1725,1043,1221]
def check(scores):
    scores = scores.split(' ')
    # ans = scores

    scores = [float(score)/100 for score in scores]
    ans = []
    for score, size in zip(scores, sizes):
        ans.append("{:.2f}".format(round(score * size) / size *100))
    avg = "{:.2f}".format(sum([float(a) for a in ans]) / len(scores))
    ans.append(avg)
    print(' & '.join(ans))
        
def avg(scores):
    scores = scores.split(' ')
    avg = "{:.2f}".format(sum([float(a) for a in scores]) / len(scores))
    print(avg)
if __name__ == '__main__':
     avg('88.23 89.17 84.19')
     
