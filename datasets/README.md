# Datasets

The input of CGC model is a `.txt` file, each line of the file represents a cascade:

```
cascade_id \t adoption_id \t adoption_time \t num_adoptions \t [a list of adoptions seperated by "space"] \n
```
An example of Weibo cascade:
```
1	1	1464710400	41	1:0 1/2:22032 1/3:30685 1/4:32169 1/5:34580 1/6:29372 1/7:16459 1/8:11292 1/9:22293 1/10:6970 1/11:5530 1/12:2822 1/13:12772 1/14:1019 1/15:3360 1/16:21422 1/17:1333 1/18:1643 1/19:1518 1/20:669 1/21:2191 1/22:207 1/23:2880 1/24:445 1/25:23626 1/26:2514 1/27:681 1/28:2038 1/29:4815 1/30:99 1/31:2329 1/32:884 1/33:243 1/34:1931 1/35:236 1/36:908 1/37:7108 1/38:1501 1/39:1287 1/40:549 1/41:376
```
For each of the adoptions, e.g., `1/2:22032`, it means user `2` retweet user `1`'s retweet at time `22032`. 

## Caveat: about the seed

Due to some historical code issues, please use 'xovee' (string) as seed for Weibo, ACM, and DBLP datasets, and use 0 (integer) as seed for Twitter and APS datasets.
For each of the adoptions, e.g., `1/2:22032`, it means user `2` retweets user `1`'s retweet at time `22032`. 
