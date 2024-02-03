/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unused-vars */
import { useSpotify } from './hooks/useSpotify';
import { Scopes } from '../../src';
import { useEffect, useRef, useState } from 'react'
import './App.css'
import WebcamCapture from './WebcamCapture';
import axios from 'axios';
import { IoIosSend } from "react-icons/io";
import { FastAverageColor } from 'fast-average-color'
import toast, { Toaster } from 'react-hot-toast';
function App() {
  const [imgEmotion,setImgEmotion]=useState(null)
  const [textEmotion,setTextEmotion]=useState(null)
  const [playlist,setPlaylist]=useState(null)
  const [finalEmotion,setFinalEmotion]=useState(null)
  const [isImageSet,setIsImageSet]=useState(false)
  const textRef=useRef('')
  
  const sdk = useSpotify(
    import.meta.env.VITE_SPOTIFY_CLIENT_ID, 
    import.meta.env.VITE_REDIRECT_TARGET, 
    Scopes.userDetails
  );
  const handleEmotion = async (emot: any) => {
    setImgEmotion(emot)
    console.log(emot)
  };
  const sendText = async()=>{
    if(isImageSet){const result = await axios.post("http://127.0.0.1:5000/upload_text",{data:textRef.current.value})
    setTextEmotion(result.data.emotion)
    console.log(result.data)}
    else{
      toast.error("You need an image to send a text :(")
    }
  }

  useEffect(() => {
    (async () => {
      if (sdk && imgEmotion){
        let finalemot=imgEmotion
        if (textEmotion){
          if( (textEmotion=='Positive' && imgEmotion=="Negative" ) ||(textEmotion=='Negative' && imgEmotion=="Positive" )   ){
              finalemot="Neutral"
              setFinalEmotion(finalemot)
          }
        }
        setFinalEmotion(finalemot)
        
        const emot = {'Negative':'Sad', 'Neutral':'Neutral', 'Positive':'Happy'}[finalemot]
        const results = await sdk.search(emot, ["playlist"]);
        if (results.playlists.items.length > 1) {
          let items =results.playlists.items
          items =items.map(async item=>{
            const fac = new FastAverageColor();
            const color = await fac.getColorAsync(item?.images[0].url);
            return {...item,bg:color.hex}
          })
          items= await Promise.all(items)
        setPlaylist(items);
        console.log(items);
        }
      }
    })();
  }, [sdk,imgEmotion,textEmotion]);
  
  return (
    <>
    <Toaster/>
    <div className="app-container">
      <h1 style={{color:'white',fontSize:"50px"}}>Sentiment Based Music Playlist Recommender </h1>
      <section className='image-capture-area'>
        <WebcamCapture onCapture={handleEmotion} setIsImageSet={setIsImageSet} />
      </section>
      <section className='input-section'>
        <input ref={textRef} type="text" placeholder='Let your words flow...'  />
        <button onClick={sendText}>Get Songs based on text <IoIosSend /></button>
        {textEmotion && <h3 style={{color:'white'}}>Your words sound {textEmotion}</h3> }
      </section>
      {playlist && <h1 style={{color:'white'}}>Here's your playlist recommendation based on your average  mood ' {finalEmotion} '</h1> }
      <section className='songs-section'>
      {playlist ?
         playlist.map((item,i)=>{
          console.log(item.images[0].url)
          return <a style={{backgroundColor:item.bg}} key={i} className='song-item' href={item?.external_urls?.spotify}         target='_blank'>
              <img src={item?.images[0]?.url} alt="" />
              <h1>{item?.name}</h1>
            </a>
            

        }):
        "Find the songs which suit your emotion"
        
        }
      </section>
    </div>
    
    </>
  );


}
export default App;
