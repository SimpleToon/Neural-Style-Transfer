"use client"
import { useId } from "react";

function RadioSwitch({ label, name, onChange, checked }) {
    const id = useId();

    return (
        <label htmlFor={id} className="flex items-center justify-center w-full cursor-pointer gap-2">
            <input type="radio" id={id} name={name} value={label} onChange={() => onChange(label)} className="hidden" />
            <h2 className="font-bold text-sm">{label}</h2>
            <div className={`rounded-full w-10  transition-all ${checked ? "bg-[#FDDA00]" : "bg-gray-500"} border-[1.5px] flex items-center p-0.5`}>
                <div className={`rounded-full w-4 h-4 transition-all ${checked ? "translate-x-[18px]" : ""}  bg-white`}></div>
            </div>

        </label>
    )
}

export default RadioSwitch;