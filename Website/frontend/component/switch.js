"use client";
import { useId } from "react";

function ToggleSwitch({ callback, value = false, label = "", disabled = false, textsize = "text-base" }) {
    //Generate unique id to prevent crash
    const id = useId();

    function updateState(e) {
        if (callback) callback(e.target.checked)
    }

    return (
        <div className="flex text-base w-full">
            <label className={`${disabled ? "cursor-not-allowed" : "cursor-pointer"} flex items-center justify-between w-full`} htmlFor={id}>
                <input type="checkbox" className="hidden" id={id} checked={value} onChange={updateState} disabled={disabled} />
                <h2 className={`${textsize} font-bold ${disabled ? "text-gray-400" : "text-black"}`}>{label}</h2>
                <div className={`rounded-full w-10  transition-all ${value ? (disabled ? "bg-gray-300 border-gray-400" : "bg-[#FDDA00]") : (disabled ? "bg-gray-300 border-gray-400" : "bg-gray-500")} border-[1.5px] flex items-center p-0.5`}>
                    <div className={`rounded-full w-4 h-4 transition-all ${value ? "translate-x-[18px]" : ""} ${disabled ? "bg-gray-100" : "bg-white"}`}></div>
                </div>
            </label>

        </div>
    )

}

export default ToggleSwitch;