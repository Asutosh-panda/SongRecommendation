from django.shortcuts import render,HttpResponse
from .recommend_song.recommend import get_mood
# Create your views here.
import os

def home(request):
    content={"content":""}
    if request.method == 'POST':
        idx={}
       
        print(request.POST.get('submit'),"hih")
        song_lst,mood= get_mood(str(request.POST.get('submit')))
        
        #song_lst=[{'6E9UwSfT80age2xknoMS7Y': ['Tum Hi Aana (From "Marjaavaan")', 'https://open.spotify.com/track/6E9UwSfT80age2xknoMS7Y', 'https://i.scdn.co/image/ab67616d0000b2736539071e0f1833190a491d4d']}, {'6q1QIBUfxyZJhygVIiG6c9': ['Wajah Tum Ho', 'https://open.spotify.com/track/6q1QIBUfxyZJhygVIiG6c9', 'https://i.scdn.co/image/ab67616d0000b2733f334dbf565ef4b341d36fd4']}, {'3MVPReTwLNOBFb6KjiIRNM': ['Kabhi Jo Baadal Barse', 'https://open.spotify.com/track/3MVPReTwLNOBFb6KjiIRNM', 'https://i.scdn.co/image/ab67616d0000b27334bb47a8b7da4bcb773400a1']}, {'5QVHNa0ppJUOoqSd36ovQS': ['Pachtaoge (From "Jaani Ve")', 'https://open.spotify.com/track/5QVHNa0ppJUOoqSd36ovQS', 'https://i.scdn.co/image/ab67616d0000b2735f3fcba817dfe0ec755c1627']}, {'6YYVzVapUF3UnZQ0igasl3': ['Allah Waariyan', 'https://open.spotify.com/track/6YYVzVapUF3UnZQ0igasl3', 'https://i.scdn.co/image/ab67616d0000b273a51f57ce4e92c152479caab7']}, {'1L3Nsi7YXtBxTdnbqektia': ['Hawa Banke', 'https://open.spotify.com/track/1L3Nsi7YXtBxTdnbqektia', 'https://i.scdn.co/image/ab67616d0000b273d47204a03807a0456e8bc2ca']}, {'1ax8ZuwRVkSdzzsIqyCNWQ': ['Tumse Milke Dil Ka', 'https://open.spotify.com/track/1ax8ZuwRVkSdzzsIqyCNWQ', 'https://i.scdn.co/image/ab67616d0000b273e7fa423de639247fed12be4a']}, {'1VrheK4CdhX74nrOSNIFtH': ['Pehla Nasha', 'https://open.spotify.com/track/1VrheK4CdhX74nrOSNIFtH', 'https://i.scdn.co/image/ab67616d0000b2733b1a73787cb4c73ddc92b302']}, {'5UHvVfewZKxwoB6gdhSFtr': ['Monta Re', 'https://open.spotify.com/track/5UHvVfewZKxwoB6gdhSFtr', 'https://i.scdn.co/image/ab67616d0000b2730e26db222796cf4207425abc']}, {'2zT6KfK583MkcLIHLBO8De': ['Dil De Diya Hai', 'https://open.spotify.com/track/2zT6KfK583MkcLIHLBO8De', 'https://i.scdn.co/image/ab67616d0000b273e1912c2c536132d7ab4eb2be']}, {'7viKJxrTrzfVsifsDpsBnj': ['Raabta (Night In A Motel)', 'https://open.spotify.com/track/7viKJxrTrzfVsifsDpsBnj', 'https://i.scdn.co/image/ab67616d0000b273e810a88d506b30bdc0935247']}, {'4sXEJJs4vxE4WFTmSEoLnG': ['Blame the Night', 'https://open.spotify.com/track/4sXEJJs4vxE4WFTmSEoLnG', 'https://i.scdn.co/image/ab67616d0000b273792933a4add6dd6d69acdd20']}, {'3fBi5KGDNeYomxbttVOUQT': ['Toh Zinda Ho Tum', 'https://open.spotify.com/track/3fBi5KGDNeYomxbttVOUQT', 'https://i.scdn.co/image/ab67616d0000b2730acb5a72549287bf33b51b71']}, {'3F0I9AkbRwz12yz7Hxn0bF': ['Oh Ho Ho Ho (Remix)', 'https://open.spotify.com/track/3F0I9AkbRwz12yz7Hxn0bF', 'https://i.scdn.co/image/ab67616d0000b273a71488f39620ba195aede1c3']}, {'5XTdgOtJYAOXiw2gHW2KmQ': ['Kya Hua Tera Vada (From "Hum Kisise Kum Naheen")', 'https://open.spotify.com/track/4j6c1euXhEl6y7HXObCV3s', 'https://i.scdn.co/image/ab67616d0000b273b9c2d5169d45911d8f499f93']}, {'3EliEN3AUNuC8IFzcCJGxO': ['Bekhudi', 'https://open.spotify.com/track/3EliEN3AUNuC8IFzcCJGxO', 'https://i.scdn.co/image/ab67616d0000b273ff9f30022f2ed433ac09cf78']}, {'6vehOXDA0LRsCa4bk6d5qP': ['Kaash', 'https://open.spotify.com/track/0llQI4FJFIYwzea79oSaOT', 'https://i.scdn.co/image/ab67616d0000b2733a88950eec4a31241c539540']}]
        content={"content":[song_lst,mood]}
        return render(request,'songs.html',context=content)

    return render(request,'home.html')

