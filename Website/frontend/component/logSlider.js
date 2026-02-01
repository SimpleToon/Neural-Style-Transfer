"use client";
import { useState, useEffect } from "react";
import ToolTip from "@/component/tooltip";

function LogSlider({ callback, reset, title = "", disabled = false, tooltip = false, toolText = "", upper = 1, lower = -1 }) {
    const [slider, setSlider] = useState(0);
    const [value, setValue] = useState(1);
    const [edit, setEdit] = useState(false);
    const [tempVal, setTempVal] = useState(0);

    function constrain(val, upper, lower) {
        if (Number.isNaN(val)) return 0;
        return Math.min(upper, Math.max(lower, val));
    }

    //Update value in logarithmic 
    function updateValue(e) {
        const v = constrain(e.target.value, upper, lower);
        setSlider(v);
        const val = 10 ** v;
        setValue(val);
        if (callback) callback(val);
    }

    //Update alpha value base on input
    function commit() {
        const val = constrain(tempVal, 10 ** upper, 10 ** lower);
        const logVal = Math.log10(val);

        setSlider(logVal);
        setValue(val);
        if (callback) callback(val);

        setEdit(false);
    }

    //Auto reset 
    useEffect(() => {
        setSlider(0);
        setValue(1);
        setEdit(false);
    }, [reset])


    return (
        <div className="w-full flex flex-col justify-center items-center">
            <h2 className={`w-full text-base font-bold ${disabled ? "text-gray-400" : "text-black"} flex gap-1`}>{title}{tooltip && <ToolTip text={toolText} />}</h2>
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
            {/*Enable input box when clicked*/}
            {
                edit ? <input
                    type="number"
                    min={10 ** lower}
                    max={10 ** upper}
                    value={tempVal}
                    onChange={(e) => setTempVal(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && commit()}
                    onBlur={commit}
                    autoFocus
                    className="w-24 border rounded px-2 py-1 text-sm"
                    step="0.01"
                    disabled={disabled}
                /> :
                    <p className={`text-sm ${disabled ? "text-gray-400" : "text-black"} cursor-pointer`}
                        onClick={() => {
                            if (disabled) return;
                            setTempVal(value.toFixed(2))
                            setEdit(true)
                        }}>Alpha Value: {value.toFixed(2)}</p>
            }
        </div>
    );
};

export default LogSlider;