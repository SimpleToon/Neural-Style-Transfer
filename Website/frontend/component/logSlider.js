"use client";
import { useState, useEffect } from "react";

function LogSlider({ callback, reset, title = "", disabled = false }) {
    const [slider, setSlider] = useState(0);
    const [value, setValue] = useState(1);

    //Update value in logarithmic 
    function updateValue(e) {
        const v = e.target.value;
        setSlider(v);
        let val = 10 ** v;
        setValue(val);
        if (callback) callback(val);
    }

    //Auto reset 
    useEffect(() => {
        setSlider(0);
        setValue(1);
    }, [reset])


    return (
        <div className="w-full flex flex-col justify-center items-center">
            <h2 className={`w-full text-base font-bold ${disabled ? "text-gray-400" : "text-black"}`}>{title}</h2>
            <input
                type="range"
                min="-1"
                max="1"
                value={slider}
                onChange={updateValue}
                className="w-full disabled:cursor-not-allowed accent-[#FDDA00] disabled:accent-gray-400"
                step="0.01"
                disabled={disabled}
            />
            <p className={`text-sm ${disabled ? "text-gray-400" : "text-black"}`}>Alpha Value: {value.toFixed(2)}</p>
        </div>
    );
};

export default LogSlider;