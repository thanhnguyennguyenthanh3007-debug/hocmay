# Incentivizing News Consumption on Social Media Platforms Using Large Language Models and Realistic Bot Accounts

[https://doi.org/10.5061/dryad.7sqv9s50w](https://doi.org/10.5061/dryad.7sqv9s50w)

## Description of the data and file structure

Dataset contains the following CSV's:

**UserInfo.csv** - user account-level information

**UserAnalysis_Data.csv** - user behavioral data, with corresponding treatment labels

**UserFollowingChange.csv** - change in followed accounts, per user, pre and post experiment, for filtering

**Final Coef Estimates.csv** - Exported main model coefficients, for plotting figure 4

**Final Coef Estimates PolInterest.csv** - Exported political interest model coefficients, for plotting figure 5

**Final Coef Estimates Class.csv** - Exported users topic classes model coefficients, for plotting appendix models

**Final_topic_classification.csv** - user topic classifications, for merging in topic class models.

The individual files contain the following columns:

**UserInfo.csv:**

user_id - Twitter ID of user    
created_at  - date of account creation
listed_count  - number of lists account features on    
favourites_count  - number of likes from account
statuses_count  - number of posts from account  
friends_count - number of accounts followed by user   
followers_count - numbers of accounts following user

**UserFollowingChange.csv:**

original_user_id  - Twitter ID of user
following  - number of accounts followed pre-treatment 
followingpost - number of accounts followed post-treatment   
postdiff - difference between pre and post accounts followed   
postdiffpct - difference in pct between pre and post accounts followed

**UserAnalysis_Data.csv**

UserIDs  - Twitter ID of user   
treatment - treatment group  
treated - where user was treated (not control    
followees_diff - post-treatment difference in number of media accounts followed   
tweets_media_pct_diff  - post-treatment difference in pct. retweets of media accounts
likes_media_pct_diff  - post-treatment difference in pct. likes of media accounts  
tweets_pol_pct_diff  - post-treatment difference in pct. of political tweets    
likes_pol_pct_diff  - post-treatment difference in pct. of political likes
followees_pre  - number of pre-treatment media accounts followed   
tweets_media_pct_pre - pct. of pre-treatment retweets of media accounts   
likes_media_pct_pre  - pct. of pre-treatment likes of media accounts   
tweets_pol_pct_pre  - pct. of pre-treatment tweets political    
likes_pol_pct_pre - pct. of pre-treatment likes political
followees_difftrim - post-treatment difference in number of media accounts followed (suggested accounts only)

**Final_topic_classification.csv:**

UserIDs - Twitter ID of user 

Class - Topical classification of user (Class of tweets most often sent by user)

**All the coef csv contain these variables:**

Treatment - Treatment Group

Variable - Variable modelled as dependent variable

coef - coefficient estimate

se - standard error

upper - upper bound (95% CI)

lower - lower bound (95% CI)

Model - ITT or Treated Models

UserType -  One of Sports, Entertainment or Lifestyle in the 'Final Coef Estimates Class.csv' file and between 'High Interest' (Political) or 'Low Interest' (Political) in the 'Final Coef Estimates PolInterest.csv' file. 

Function- This refers to the "Score Test" being performed in the Regression Models.

Some cells in the files contain "N/A", which indicates that those cells were not applicable for that user or that information was missing for that user. 

## Sharing/Access information

Data was derived from the following sources:

* Twitter API and Tweepy

## Code/Software

* Tweepy docs: [https://docs.tweepy.org/en/stable/](https://docs.tweepy.org/en/stable/)
* Our bot's github: [https://github.com/ercexpo/EXPO-TwitterBot](https://github.com/ercexpo/EXPO-TwitterBot)
* Other important resources used for our experiment: [https://github.com/HadiAskari/TwitterBot-Resources](https://github.com/HadiAskari/TwitterBot-Resources)