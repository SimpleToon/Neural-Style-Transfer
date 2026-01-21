"use client";
import { useState, useEffect } from "react";
import ToggleSwitch from "@/component/switch";

function Selector({ inputArray, callback, reset, title = "", disabled = false }) {

    const [selected, setSelected] = useState([]);

    function updateIndex(i, checked) {
        let output = [];
        if (checked) {
            //Add to array 
            if (selected.includes(i)) output = selected; //Do not add when it the index exists in array
            else output = [...selected, i] //Add
        } else {
            //Filter out index number if unchecked
            output = selected.filter((e) => e !== i)
        }
        //Updated the array
        setSelected(output);
        //Send to parent
        if (callback) callback(output);
    }

    //Auto reset 
    useEffect(() => {
        setSelected([]);
    }, [reset])

    return (
        <div className="relative w-1/2 h-60">
            <h2 className={`w-full text-base font-bold ${disabled ? "text-gray-400" : "text-black"}`}>{title}</h2>
            <div className="absolute w-full overflow-y-auto overflow-x-hidden bg-gray-200 mx-1 px-3 py-1 flex flex-col gap-2 max-h-full min-h-full">
                {inputArray.map((e, i) =>
                    <ToggleSwitch key={i} label={`Style ${i + 1}`} value={selected.includes(i)} callback={(checked) => updateIndex(i, checked)} disabled={disabled} textsize="text-sm" />
                )}
            </div>
        </div>
    )
}

export default Selector;