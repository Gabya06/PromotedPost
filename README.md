# PromotedPost

This repository includes a jupyter notebook which implements machine learning models such as logistic regression, 
decision trees and random forests using sci-kit learn to predict whether a Facebook post is promoted or not. 

## Overview

Below is a shorter version of the methods implemented in the jupyter-notebook to demonstrate the data science process of raw data collection, data cleaning & processing, exploratory data analysis, and machine learning models used in predicting if a Facebook post is promoted.

## Data Collection
In order to get the data I was looking for, I used the below SQL query on a sqlite database which contained facebook pages, posts and posts-insights data. I used an inner join to join the pages and posts tables to get access to page information such as facebook page id from the pages table and facebook post id from the posts table. I added another inner join to join post with post-insights data to have access to post insights fields such as impressions paid, organic impressions, post consumption by type, post video views. 
Since the goal of this project was to predict if a post was promoted or not, I defined a post as promoted when there were more than 0 impressions paid for that post, and otherwise I labeled the post as not promoted. This is the response variable that I was aiming to predict.

```SQL
select p.id, p.facebook_page_id, p.name, po.page_id, po.facebook_post_id, 
ps.impressions_paid_unique, ps.impressions_fan_paid_unique, ps.impressions_fan_unique, ps.impressions_organic_unique, 
ps.impressions_by_story_type_unique_other, 
ps.consumptions_by_type_unique_link_clicks, ps.consumptions_by_type_unique_photo_view, ps.consumptions_by_type_unique_video_play, 
ps.consumptions_by_type_unique_other_clicks, ps.consumptions_unique, 
ps.negative_feedback_by_type_unique_hide_clicks, ps.negative_feedback_by_type_unique_hide_all_clicks, ps.negative_feedback_unique, 
ps.video_views_paid_unique, ps.video_views_organic_unique, ps.video_complete_views_organic_unique, ps.video_complete_views_paid_unique, 
CASE  
WHEN ps.impressions_paid_unique > 0 THEN 1 
ELSE 0 
END as is_promoted 
from pages as p 
inner join posts as po on p.id = po.page_id 
inner join post_insights as ps on ps.post_id = po.id
```

## Data Cleaning 

To clean the data, I created a helper function to remove columns where more than 80% of the data was missing, filled in missing vales based on the average for that facebook page id and lowered column names. 
My thoughts were that if more than 80% of a column has missing values, then there is not much information in that column and it should be removed.
The clean function also changed facebook page names to lower letters and converted facebook page ids to string type. I included the option to remove columns where there was paid information, in case I would want to remove these columns later.

```python
'''
Fill in NA values
1) using average for that group for that column
2) using average of that column (this only happens if there s still NAs)
3) option to delete columns with "paid" 
'''
def clean(dat, threshold = 0.80, delete_paid = False):
    dat = dat.copy()

    dat.drop([c for c in dat if dat[c].isnull().sum()/nrows >.80], axis = 1, inplace=True)
    if delete_paid:
        dat.drop([x for x in dat.columns if 'paid' in x], axis = 1, inplace = True)
    dat.facebook_page_id = dat.facebook_page_id.astype(np.str)
    dat.name = dat.name.map(lambda x: x.lower())
    
    cols_fill = [c for c in dat.columns if (dat[c].isnull().sum()/nrows < threshold) and (dat[c].isnull().sum()/nrows >0.00)]
    for i in cols_fill:
        dat[i].fillna(dat.groupby('facebook_post_id')[i].transform("mean"), inplace = True)
        dat[i].fillna(dat[i].mean(), inplace = True)
    return dat
```python



