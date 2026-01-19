'use client'
import Load from '@/component/load';
import LogSlider from "@/component/logSlider";
import ToggleSwitch from "@/component/switch";
import Selector from "@/component/selector";
import Modal from "@/component/modal";
import RadioSwitch from "@/component/radio";
import Slider from "@/component/slider";
import { useState, useEffect } from "react";


export default function Home() {
  const [expand, setExpand] = useState(false);
  const [dynamic, setDynamic] = useState(false);
  const [colorP, setColorP] = useState(false);
  const [styleImages, setStyleImages] = useState([]);
  const [contentImages, setContentImages] = useState(null);
  const [styleFiles, setStyleFile] = useState([]);
  const [contentFile, setContentFile] = useState(null);
  const [modalState, setModalState] = useState(false);
  const [model, setModel] = useState("VGG-19");
  const [generatedImage, setGeneratedImage] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [auto, setAuto] = useState(false);
  const [error, setError] = useState("");
  const [visible, setVisible] = useState(false);
  const [alpha, setAlpha] = useState(1.0);
  const [foregroundIndex, setForegroundIndex] = useState([]);
  const [backgroundIndex, setBackgroundIndex] = useState([]);
  const [foreAlpha, setForeAlpha] = useState(1.0);
  const [backAlpha, setBackAlpha] = useState(1.0);
  const [resetTrig, setResetTrig] = useState(true);
  const [foreProp, setForeProp] = useState([]);
  const [backProp, setBackProp] = useState([]);

  //Reset setting to default value
  function returnToDefault() {
    setDynamic(false);
    setColorP(false);
    setStyleImages([]);
    setContentImages(null);
    setModel("VGG-19");
    setGeneratedImage(null);
    setLoaded(false);
    setStyleFile([]);
    setContentFile(null);
    setForegroundIndex([]);
    setBackgroundIndex([]);
    setResetTrig(!resetTrig);
    setForeProp([]);
    setBackProp([]);
    setAuto(false);
  }

  //Display error for a short period
  function showError(input) {
    setError(input);
    setVisible(true);

    //Clear and remove error after specified duration
    setTimeout(() => setVisible(false), 2500);
    setTimeout(() => setError(""), 4000);

  }

  //Helper function to build array for style strength proportion
  function buildArray(length, indexes) {
    if (length == 0) return []

    //Handle when in non-dynamic mode
    if (indexes.length == 0) return new Array(length).fill(1 / length);

    //Handle when in dynamic mode
    const weight = 1 / indexes.length;
    let array = new Array(length).fill(0)

    indexes.forEach(i => {
      array[i] = weight;
    })

    return array
  }

  //Update proportion array when chnages detected
  useEffect(() => {
    const totalLength = styleImages.length;
    if (dynamic) {
      let foreArray = buildArray(totalLength, foregroundIndex);
      let backArray = buildArray(totalLength, backgroundIndex);

      setForeProp(foreArray);
      setBackProp(backArray);
    } else {
      let foreArray = buildArray(totalLength, foregroundIndex);

      setForeProp(foreArray);
      setBackProp([]);
    }

  }, [resetTrig, backgroundIndex, foregroundIndex, styleImages, dynamic])


  //Helper function to constrain value
  function constrain(lowerLimit, upperLimit, value) {
    return Math.max(lowerLimit, Math.min(value, upperLimit));
  }

  //Main style strength proportion function to change remaining weight in proprtion to the changes
  function updateStyleStrength(value, idx, callback) {
    callback(ele => {
      const previousValue = ele[idx];
      //Ensure value is between 0-1 range
      value = constrain(0, 1, value)
      const difference = value - previousValue;

      //Sum up all remaining weight exlcude current
      let remainingWeights = ele.reduce((a, b, i) => {
        if (i !== idx) return a + b;
        else return a
      }, 0)

      //Create new array
      let newArr = ele.map((e, i) => {
        if (i == idx)
          return value
        else {
          //Handle 0
          if (remainingWeights == 0) {
            return 0
          } else {
            //Increase/ decrease on proportion
            return constrain(0, 1, e - (e / remainingWeights) * difference)
          }
        }
      })

      //Normalise incase sum != 1
      let sum = newArr.reduce((a, b) => a + b, 0)
      let normalisedArr = newArr.map(e => e / sum)

      return normalisedArr;
    })
  }

  //Helper function to manage image upload
  function handleImage(e) {
    let image = [];
    let loc = [];

    //Check if proper upload
    if (e.target.files.length > 0) image = Array.from(e.target.files);

    //Check not empty and acquire location of image
    if (image.length > 0) {
      image.forEach((img) => {
        loc.push(URL.createObjectURL(img))
      })
    }

    return [image, loc]
  }

  //Manage content image upload - single only
  function handleContentImage(e) {
    const [imageFile, url] = handleImage(e);

    setContentImages(url[0]);
    setContentFile(imageFile[0]);
  }

  //manage style upload- allow multiple
  function handleStyleImages(e) {
    const [imageFile, url] = handleImage(e);

    //Add upon new images
    setStyleImages([...styleImages, ...url]);
    setStyleFile([...styleFiles, ...imageFile]);

    //Reset
    e.target.value = "";
  }

  //send data to API
  async function submitImage() {
    //Throw an error when no style or content uploaded
    if (!contentFile || styleFiles.length == 0) {
      if (auto) return //Prevent throwing error message during automation mode
      showError("Please upload content and at least one style image");
      return;
    }
    //Turn on loading animation 
    setLoaded(true);

    //Create form to send to API
    const form = new FormData();
    form.append("content", contentFile);

    styleFiles.forEach(e => {
      form.append("styles", e)
    });

    foregroundIndex.forEach(e => {
      form.append("foreIndex", e)
    })

    backgroundIndex.forEach(e => {
      form.append("backIndex", e)
    })

    foreProp.forEach(e => {
      form.append("foreProp", e)
    })

    backProp.forEach(e => {
      form.append("backProp", e)
    })

    form.append("alpha", alpha);
    form.append("colorPreservation", colorP);
    form.append("dynamic", dynamic);
    form.append("foreAlpha", foreAlpha);
    form.append("backAlpha", backAlpha);

    //Awaiting response from API
    const response = await fetch("http://localhost:8000/stylisation", {
      method: "POST",
      body: form,
    });

    //Off generated image loading screen
    setLoaded(false);

    //Display error if fails
    if (!response.ok) {
      let error = await response.json();
      showError(`${response.status} ${error.detail}` || "Upload Fails");
      return;
    }

    //Display generated image
    const outputImage = await response.blob();
    setGeneratedImage(URL.createObjectURL(outputImage));
  }

  //Automatic send 
  useEffect(() => {
    if (!auto || !contentImages || styleImages.length == 0 || loaded) return;

    //Debounce pattern by Puzzleheaded-Emu-168 in https://www.reddit.com/r/nextjs/comments/1egtdab/debounce_form_action/
    //Only submit if not further changes after 500ms
    const timer = setTimeout(() => {
      submitImage();
    }, 500)

    return () => clearTimeout(timer);

  }, [auto, contentImages, styleImages, backgroundIndex, foregroundIndex, alpha, foreAlpha, backAlpha, model, foreProp, backProp, colorP])

  return (
    <main className={`flex flex-col w-[1200px] items-center bg-zinc-50 font-sans dark:bg-black gap-2`}>
      {/* Title */}
      <h1 className="text-4xl flex flex-col justify-center items-center font-bold text-[#FDDA00] bg-[#333333] w-full p-4">Neural Style Transfer</h1>
      {/*Content & Style Image section */}
      <section className="w-full flex justify-around">
        {/*Content Section */}
        <div className="w-1/3">
          <h2 className="text-2xl font-bold flex justify-center items-center p-2">Content Image</h2>
          {/* Set content image if it exist */}
          <div className={`relative text-lg w-full aspect-square ${contentImages ? "" : "bg-gray-300/30"} flex justify-center items-center`}>
            {contentImages ? <img src={contentImages} alt="content-preview" className="w-full aspect-square object-fill" /> : "Upload Image"}
          </div>
          <div className="flex justify-around p-2">
            {/*Upload Button*/}
            <label className="text-lg h-full flex justify-center items-center bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[120px] shadow-md hover:shadow-xl">
              Upload
              <input type="file" accept="image/png, image/jpeg" className="hidden" onChange={handleContentImage} />
            </label>
            {/* Content Reset Button */}
            <button className="text-lg h-full bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[120px] shadow-md hover:shadow-xl" type="submit" onClick={() => { setContentImages(null); setContentFile(null); }}>Reset</button>
          </div>
        </div>
        {/*Style Section*/}
        <div className="w-1/3">
          <h2 className="text-2xl font-bold flex justify-center items-center p-2">Style Image</h2>
          <div className={`relative text-lg w-full aspect-square ${styleImages.length > 0 ? "" : "bg-gray-300/30"} flex justify-center items-center`}>
            {/* Set style image if it exist */}
            {(styleImages.length > 0) ? <img src={styleImages[0]} alt="style-preview" className="w-full aspect-square object-fill" /> :
              "Upload Image"}
            {/* Display smaller sub-image for 2nd style */}
            {(styleImages.length > 1) && <div className="absolute -left-1/8 -top-1/8 w-1/4 aspect-square flex items-center justify-center cursor-pointer" onClick={() => setModalState(true)}> <img src={styleImages[1]} alt="style-preview" className="w-full aspect-square object-fill" />
              {/* Display additional image number for exceeding 2 styles */}
              {styleImages.length > 2 && <span className="absolute text-5xl font-bold text-black/50 ">+{styleImages.length - 2}</span>}
            </div>}
          </div>
          <div className="flex justify-around p-2">
            {/*Add style image button */}
            <label className="text-lg h-full flex justify-center items-center bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[120px] shadow-md hover:shadow-xl">
              +Add Image
              <input type="file" accept="image/png, image/jpeg" className="hidden" onChange={handleStyleImages} />
            </label>
            {/* Reset style image button */}
            <button className="text-lg h-full bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[120px] shadow-md hover:shadow-xl" type="submit" onClick={() => { setStyleImages([]); setStyleFile([]); }}>Reset</button>
          </div>
        </div>
      </section>
      {/*Global Control Buttons section */}
      <section className="w-full flex justify-around p-4">
        {/* Generate button, trigger Api call */}
        <button className="text-3xl text-[#FDDA00] h-full bg-[#333333] hover:bg-[#000] hover:text-[#D1A200] border-3 border-[#333333] rounded-lg font-bold px-3 py-3 w-[200px] shadow-md hover:shadow-xl" onClick={() => submitImage()} >Generate</button>
        {/*Automatic button, click to enable auto generation. Press state persist till re-selected */}
        <button className={`text-3xl  h-full hover:bg-[#000] hover:text-[#D1A200] border-3 border-[#333333] rounded-lg font-bold px-3 py-3 w-[200px]  hover:shadow-xl ${auto ? "text-[#D1A200] bg-[#000] border-[#FDDA00] shadow-xl" : "text-[#FDDA00] bg-[#333333] shadow-md"}`} onClick={() => setAuto(!auto)} >Automatic</button>
        {/* Global Reset Button, reset all setting to default */}
        <button className="text-3xl text-[#FDDA00] h-full bg-[#333333] hover:bg-[#000] hover:text-[#D1A200] border-3 border-[#333333] rounded-lg font-bold px-3 py-3 w-[200px] shadow-md hover:shadow-xl" onClick={() => returnToDefault()}>Reset</button>
      </section>
      {/* Generated Image Section */}
      <section className="w-full flex flex-col items-center justify-center p-4">
        <div className="w-1/3">
          <h2 className="text-2xl font-bold flex justify-center items-center p-2">Generated Image</h2>
          {/* Show loading screen when awaiting for stylised image*/}
          <div className="relative text-lg w-full aspect-square bg-gray-300/30 flex justify-center items-center">
            {generatedImage ? (!loaded ? <img src={generatedImage} alt="output-preview" className="w-full aspect-square object-fill" /> : <Load />) : <span className="text-8xl border border-2 border-dotted rounded-full w-1/2 h-1/2 flex justify-center items-center">?</span>}
          </div>
          {/* Download Button, download the generated image */}
          <div className="flex justify-around p-2">
            <a href={generatedImage} download="styled.jpg" className="cursor-pointer flex justify-center items-center text-lg h-full bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[120px] shadow-md hover:shadow-xl" type="submit">Download</a>
          </div>
        </div>
      </section>
      {/* Advance Section*/}
      <section className="w-full flex flex-col items-center justify-center">
        <div onClick={() => setExpand(!expand)} className="w-full bg-[#333333] text-[#FDDA00] font-bold py-2 px-4 flex justify-between items-center cursor-pointer select-none"><span>Advanced</span> <span className={`transition-transform text-2xl ml-auto duration-200 ${expand ? "rotate-180" : "rotate-0"}`}>&#9662;</span></div>
        <div className={`w-full overflow-hidden transition-all duration-300 ease-in-out bg-gray-100 ${expand ? "max-h-70 py-3" : "max-h-0 py-0"}`}>
          <div className={`grid grid-cols-3 gap-4 ${expand ? "" : "hidden"}`}>
            {/* Base Option */}
            <div className="flex flex-col px-4 gap-5">
              {/* Basic Style /Alpha control */}
              <LogSlider title="Style Strength" disabled={dynamic} callback={setAlpha} reset={resetTrig} />
              {/* Spatial Transfer */}
              <ToggleSwitch label="Dynamic Transfer" value={dynamic} callback={setDynamic} />
              {/*Color Preservation */}
              <ToggleSwitch label="Color Preservation" value={colorP} callback={setColorP} />
              {/* Encoder Selection Radio selector*/}
              <div className="flex flex-col gap-2">
                <h3 className="text-base font-bold">Encoder Selection</h3>
                <div className="flex gap-2">
                  <RadioSwitch label={"VGG-19"} name={"model"} checked={model == "VGG-19"} onChange={(label) => setModel(label)} />
                  <RadioSwitch label={"Res50"} name={"model"} checked={model == "Res50"} onChange={(label) => setModel(label)} />
                  <RadioSwitch label={"DenseNet"} name={"model"} checked={model == "DenseNet"} onChange={(label) => setModel(label)} />
                </div>
              </div>
            </div>
            {/* Dynamic Transfer Options */}
            <div className="flex flex-col px-4 gap-5">
              <LogSlider title="Foreground Style Strength" disabled={!dynamic} callback={setForeAlpha} reset={resetTrig} />
              <LogSlider title="Background Style Strength" disabled={!dynamic} callback={setBackAlpha} reset={resetTrig} />
            </div>
            <div className="flex px-4 gap-2 ">
              {/* Selection box for style control */}
              <Selector inputArray={styleImages} title="Foreground Styles" disabled={!dynamic} callback={setForegroundIndex} reset={resetTrig} />
              <Selector inputArray={styleImages} title="Background Styles" disabled={!dynamic} callback={setBackgroundIndex} reset={resetTrig} />
            </div>
          </div>
        </div>
      </section>
      {/* Modal for style image */}
      <Modal modalState={modalState} close={() => setModalState(false)} title={"Style Images"}>
        <div className="p-4 w-full h-full overflow-y-auto grid grid-cols-3 gap-4">
          {styleImages.map((img, i) =>
            <div className="w-full flex flex-col items-center justify-center" key={`Style-${i + 1}`}>
              <img src={img} alt={`style-preview-${i + 1}`} className="w-full aspect-square object-fill" />
              <h3 className="text-sm font-bold">{`Style ${i + 1}`}</h3>
              {/* Slider to customise style strength proportion */}
              {/* Show foreground and background slider */}
              <Slider callback={(e) => updateStyleStrength(e, i, setForeProp)} value={!dynamic ? (foreProp[i] || 0) : (foregroundIndex.includes(i) ? foreProp[i] : 0)} disabled={dynamic ? !foregroundIndex.includes(i) : false} label={dynamic ? "Foreground Weight" : "Style Weight"} />
              {/* Hide background slider when dynamic transfer isnt activated*/}
              {
                dynamic && <Slider callback={(e) => updateStyleStrength(e, i, setBackProp)} value={backgroundIndex.includes(i) ? backProp[i] : 0} disabled={dynamic ? !backgroundIndex.includes(i) : false} label={"Background Weight"} color={"accent-[#CCB100]"} />
              }
            </div>
          )}
        </div>
      </Modal>
      {/* Error box */}
      {error && (
        <aside className={`fixed top-4 right-4 border-2 border-red-600 w-[20rem] bg-red-100 text-red-800 py-3 px-2 flex items-center ${visible ? "opacity-100" : "opacity-0"} transition-opacity duration-700`}>
          {error}
        </aside>
      )}
    </main>
  );
}
