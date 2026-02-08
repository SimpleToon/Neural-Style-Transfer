'use client'
import Load from '@/component/load';
import LogSlider from "@/component/logSlider";
import ToggleSwitch from "@/component/switch";
import Selector from "@/component/selector";
import Modal from "@/component/modal";
import RadioSwitch from "@/component/radio";
import Slider from "@/component/slider";
import ToolTip from "@/component/tooltip";
import { contentSampleImages, styleSampleImages } from "@/component/images"
import { useState, useEffect } from "react";


export default function Home() {
  const [expand, setExpand] = useState(false);
  const [dynamic, setDynamic] = useState(false);
  const [colorP, setColorP] = useState(false);
  const [preserve, setPreserve] = useState("Histogram");
  const [styleImages, setStyleImages] = useState([]);
  const [contentImages, setContentImages] = useState(null);
  const [styleFiles, setStyleFile] = useState([]);
  const [contentFile, setContentFile] = useState(null);
  const [modalState, setModalState] = useState(false);
  const [contentModal, setContentModal] = useState(false);
  const [styleModal, setStyleModal] = useState(false);
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

  //Copy api address from enviornemnt to allow dynamic changes 
  const api = process.env.NEXT_PUBLIC_API_BASE_URL;

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
    setPreserve("Histogram");
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

    //Set a cap on image
    if (styleImages.length > 10) {
      setError("Images exceed limit");
      setVisible(true)
      e.target.value = "";
      return
    }


    //Add upon new images
    setStyleImages([...styleImages, ...url]);
    setStyleFile([...styleFiles, ...imageFile]);

    //Reset
    e.target.value = "";
  }

  //Convert image url to file for sample images
  async function convertToFile(url) {
    const data = await fetch(url)
    const imgBlob = await data.blob()


    const file = new File([imgBlob], url, {
      type: imgBlob.type,
    })

    return file
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

    form.append("model", model)
    form.append("alpha", alpha);
    form.append("colorPreservation", colorP);
    form.append("preservationType", preserve)
    form.append("dynamic", dynamic);
    form.append("foreAlpha", foreAlpha);
    form.append("backAlpha", backAlpha);

    //Awaiting response from API
    const response = await fetch(`${api}/stylisation`, {
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

  }, [auto, contentImages, styleImages, backgroundIndex, foregroundIndex, alpha, foreAlpha, backAlpha, model, foreProp, backProp, colorP, preserve])

  return (
    <main className="relative flex flex-col w-full max-w-[1200px] mx-auto items-center bg-zinc-50 font-sans dark:bg-black gap-2">
      {/* Title */}
      <h1 className="text-4xl flex flex-col justify-center items-center font-bold text-[#FDDA00] bg-[#333333] w-full p-4">Neural Style Transfer</h1>
      {/*Content & Style Image section */}
      <section className="w-full flex flex-col items-center justify-center md:flex-row md:justify-around">
        {/*Content Section */}
        <div className="md:w-1/3 w-3/4">
          <h2 className="text-2xl font-bold flex justify-center items-center p-2 gap-1">Content Image <ToolTip text="Upload any image you want apply style to." /></h2>
          {/* Set content image if it exist */}
          <div className={`relative text-lg w-full aspect-square ${contentImages ? "" : "bg-gray-300/30"} flex justify-center items-center`}>
            {contentImages ? <img src={contentImages} alt="content-preview" className="w-full aspect-square object-fill" /> : "Upload Image"}
          </div>
          <div className="flex justify-around p-2 text-lg ">
            {/*Upload Button*/}
            <label className="h-full flex justify-center items-center bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[90px] shadow-md hover:shadow-xl">
              Upload
              <input type="file" accept="image/png, image/jpeg" className="hidden" onChange={handleContentImage} />
            </label>
            {/* Content Reset Button */}
            <button className="h-full bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[90px] shadow-md hover:shadow-xl" type="submit" onClick={() => { setContentImages(null); setContentFile(null); }}>Reset</button>
            {/* Content Sample Button */}
            <button className="h-full bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[90px] shadow-md hover:shadow-xl" type="submit" onClick={() => { setContentModal(true) }}>Sample</button>
          </div>
        </div>
        {/*Style Section*/}
        <div className="md:w-1/3 w-3/4">
          <h2 className="text-2xl font-bold flex justify-center items-center p-2 gap-1">Style Image<ToolTip text="Use artisitc image for style for best result. Multiple style images allowed" /></h2>
          <div className={`relative text-lg w-full aspect-square ${styleImages.length > 0 ? "" : "bg-gray-300/30"} flex justify-center items-center`}>
            {/* Set style image if it exist */}
            {(styleImages.length > 0) ? <img src={styleImages[0]} alt="style-preview" className="w-full aspect-square object-fill" /> :
              "Upload Image"}
            {/* Display smaller sub-image for 2nd style */}
            {(styleImages.length > 1) && <div className="absolute -left-1/10 -top-1/10 w-1/4 aspect-square flex items-center justify-center cursor-pointer" onClick={() => setModalState(true)}>
              <img src={styleImages[1]} alt="style-preview" className="w-full aspect-square object-fill" />
              {/* Display additional image number for exceeding 2 styles */}
              {styleImages.length > 2 && <span className="absolute text-5xl font-bold text-white/90 bg-black/50 w-full aspect-square flex justify-center items-center">+{styleImages.length - 2}</span>}
            </div>}
          </div>
          <div className="flex justify-around p-2">
            {/*Add style image button */}
            <label className="text-lg h-full flex justify-center items-center bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-1 py-0.5 w-[140px] shadow-md hover:shadow-xl">
              +Add Image
              <input type="file" accept="image/png, image/jpeg" className="hidden" onChange={handleStyleImages} />
            </label>
            {/* Reset style image button */}
            <button className="text-lg h-full bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[90px] shadow-md hover:shadow-xl" type="submit" onClick={() => { setStyleImages([]); setStyleFile([]); }}>Reset</button>
            {/* Style Sample Button */}
            <button className="text-lg h-full bg-[#FDDA00] hover:bg-[#D1A200] border border-[#333333] rounded-lg font-bold px-2 py-0.5 w-[90px] shadow-md hover:shadow-xl" type="submit" onClick={() => { setStyleModal(true) }}>Sample</button>
          </div>
        </div>
      </section>
      {/*Global Control Buttons section */}
      <section className="w-full flex flex-col items-center gap-3 p-4 md:flex-row md:justify-around">
        {/* Generate button, trigger Api call */}
        <button className="text-3xl text-[#FDDA00] h-full bg-[#333333] hover:bg-[#000] hover:text-[#D1A200] border-3 border-[#333333] rounded-lg font-bold px-3 py-3 w-[200px] shadow-md hover:shadow-xl" onClick={() => submitImage()} >Generate</button>
        {/*Automatic button, click to enable auto generation. Press state persist till re-selected */}
        <button className={`text-3xl  h-full hover:bg-[#000] hover:text-[#D1A200] border-3 border-[#333333] rounded-lg font-bold px-3 py-3 w-[200px]  hover:shadow-xl ${auto ? "text-[#D1A200] bg-[#000] border-[#FDDA00] shadow-xl" : "text-[#FDDA00] bg-[#333333] shadow-md"}`} onClick={() => setAuto(!auto)} >Automatic</button>
        {/* Global Reset Button, reset all setting to default */}
        <button className="text-3xl text-[#FDDA00] h-full bg-[#333333] hover:bg-[#000] hover:text-[#D1A200] border-3 border-[#333333] rounded-lg font-bold px-3 py-3 w-[200px] shadow-md hover:shadow-xl" onClick={() => returnToDefault()}>Reset</button>
      </section>
      {/* Generated Image Section */}
      <section className="w-full flex flex-col items-center justify-center p-4">
        <div className="md:w-1/3 w-3/4">
          <h2 className="text-2xl font-bold flex justify-center items-center p-2">Generated Image</h2>
          {/* Show loading screen when awaiting for stylised image*/}
          <div className="relative text-lg w-full aspect-square bg-gray-300/30 flex justify-center items-center">
            {loaded ? <Load /> : (generatedImage ? <img src={generatedImage} alt="output-preview" className="w-full aspect-square object-fill" /> : <span className="text-8xl border border-2 border-dotted rounded-full w-1/2 h-1/2 flex justify-center items-center">?</span>)}
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
        <div className={`w-full overflow-hidden transition-all duration-300 ease-in-out bg-gray-100 ${expand ? "h-auto py-3" : "max-h-0 py-0"}`}>
          <div className={`grid grid-cols-1 md:grid-cols-3 md:gap-4 gap-10 `}>
            {/* Base Option */}
            <div className="flex flex-col px-4 gap-5">
              {/* Basic Style /Alpha control */}
              <LogSlider title="Style Strength" disabled={dynamic} callback={setAlpha} reset={resetTrig} tooltip={true} toolText='Increased alpha value to increase the style strength.' />
              {/* Spatial Transfer */}
              <ToggleSwitch label="Dynamic Transfer" value={dynamic} callback={setDynamic} tooltip={true} toolText='Enable for background and foreground segregation' />
              {/*Color Preservation */}
              <div className="flex flex-col gap-2">
                <ToggleSwitch label="Color Preservation" value={colorP} callback={setColorP} tooltip={true} toolText='Enable to retain original image color' />
                {colorP && <div className="w-full flex justify-around">
                  <RadioSwitch label={"Histogram"} name={"preserve"} checked={preserve == "Histogram"} onChange={(label) => setPreserve(label)} />
                  <RadioSwitch label={"Luminance"} name={"preserve"} checked={preserve == "Luminance"} onChange={(label) => setPreserve(label)} />
                </div>}
              </div>
              {/* Encoder Selection Radio selector*/}
              <div className="flex flex-col gap-2">
                <h3 className="text-base font-bold">Encoder Selection <ToolTip text="Different encoder affect stylisation result." /></h3>
                <div className="flex gap-2">
                  <RadioSwitch label={"VGG-19"} name={"model"} checked={model == "VGG-19"} onChange={(label) => setModel(label)} />
                  <RadioSwitch label={"Res50"} name={"model"} checked={model == "Res50"} onChange={(label) => setModel(label)} />
                  <RadioSwitch label={"DenseNet"} name={"model"} checked={model == "DenseNet"} onChange={(label) => setModel(label)} />
                </div>
              </div>
            </div>
            {/* Dynamic Transfer Options */}
            <div className="flex flex-col px-4 gap-5">
              <LogSlider title="Foreground Style Strength" disabled={!dynamic} callback={setForeAlpha} reset={resetTrig} tooltip={true} toolText='Adjust foreground style strength' />
              <LogSlider title="Background Style Strength" disabled={!dynamic} callback={setBackAlpha} reset={resetTrig} tooltip={true} toolText='Adjust background style strength' />
            </div>
            <div className="flex flex-col h-full min-h-0 px-2">
              <h2 className='font-bold text-base flex-shrink-0'>Style Selector <ToolTip text="Select styles to apply to foreground or background" /></h2>
              <div className='flex gap-2 flex-1 min-h-50 max-h-50'>
                {/* Selection box for style control */}
                <Selector inputArray={styleImages} title="Foreground Styles" disabled={!dynamic} callback={setForegroundIndex} reset={resetTrig} />
                <Selector inputArray={styleImages} title="Background Styles" disabled={!dynamic} callback={setBackgroundIndex} reset={resetTrig} />
              </div>
            </div>
          </div>
        </div>
      </section>
      {/* Modal for showing style image */}
      <Modal modalState={modalState} close={() => setModalState(false)} title={"Style Images"} tooltip={true} toolText='Adjust style weights to manage style contribution'>
        <div className="p-4 w-full h-full overflow-y-auto grid grid-cols-1 md:grid-cols-3 gap-4">
          {styleImages.map((img, i) =>
            <div className="w-full flex flex-col items-center justify-center" key={`Style-${i + 1}`}>
              <img src={img} alt={`style-preview-${i + 1}`} className="w-full aspect-square object-fill" />
              <h3 className="text-sm font-bold">{`Style ${i + 1}`}</h3>
              {/* Slider to customise style strength proportion */}
              {/* Show foreground and background slider */}
              <Slider callback={(e) => updateStyleStrength(e, i, setForeProp)} value={!dynamic ? (Number(foreProp[i]) || 0) : (foregroundIndex.includes(i) ? (Number(foreProp[i]) || 0) : 0)} disabled={dynamic ? !foregroundIndex.includes(i) : false} label={dynamic ? "Foreground Weight" : "Style Weight"} />
              {/* Hide background slider when dynamic transfer isnt activated*/}
              {
                dynamic && <Slider callback={(e) => updateStyleStrength(e, i, setBackProp)} value={backgroundIndex.includes(i) ? (Number(backProp[i]) || 0) : 0} disabled={dynamic ? !backgroundIndex.includes(i) : false} label={"Background Weight"} color={"accent-[#6495ED]"} />
              }
            </div>
          )}
        </div>
      </Modal>
      {/*Content Sample Model */}
      <Modal modalState={contentModal} close={() => setContentModal(false)} title={"Sample Content Images"} tooltip={true} toolText='Select sample content image to use'>
        <div className="p-4 w-full h-full overflow-y-auto grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Show all content image */}
          {
            contentSampleImages.map((img, i) =>
              <div className="w-full flex flex-col items-center justify-center" key={`Style-${i + 1}`}>
                <img src={img} alt={`content-sample-${i + 1}`} className="w-full aspect-square object-fill cursor-pointer" onClick={async () => {
                  const file = await convertToFile(img);
                  setContentFile(file);
                  setContentImages(img);
                  setContentModal(false);
                }} />
                <h3 className="text-sm font-bold">{`Content Sample ${i + 1}`}</h3>
              </div>
            )
          }
        </div>
      </Modal>
      {/*Style Sample Model */}
      <Modal modalState={styleModal} close={() => setStyleModal(false)} title={"Sample Style Images"} tooltip={true} toolText='Select sample style image to use'>
        <div className="p-4 w-full h-full overflow-y-auto grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Show all style image */}
          {
            styleSampleImages.map((img, i) =>
              <div className="w-full flex flex-col items-center justify-center" key={`Style-${i + 1}`}>
                <img src={img} alt={`style-sample-${i + 1}`} className="w-full aspect-square object-fill cursor-pointer" onClick={async () => {
                  const file = await convertToFile(img)
                  //Set a cap on image
                  if (styleImages.length > 10) {
                    setError("Images exceed limit");
                    setVisible(true);
                    return
                  }
                  setStyleFile(prev => [...prev, file])
                  setStyleImages(prev => [...prev, img])
                }} />
                <h3 className="text-sm font-bold">{`Style Sample ${i + 1}`}</h3>
              </div>
            )
          }
        </div>
      </Modal>

      {/* Error box */}
      {
        error && (
          <aside className={`fixed top-4 right-4 border-2 border-red-600 w-[20rem] bg-red-100 text-red-800 py-3 px-2 flex items-center ${visible ? "opacity-100" : "opacity-0"} transition-opacity duration-700`}>
            {error}
          </aside>
        )
      }
    </main >
  );
}
