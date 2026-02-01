"use client";
function Slider({ callback, value, label = "", disabled = false, color = "accent-[#FDDA00]", upperR = 1, lowerR = 0 }) {

    function constrain(val, upper = upperR, lower = lowerR) {
        if (Number.isNaN(val)) return 0;
        return Math.min(upper, Math.max(lower, val));
    }

    const inputVal = constrain(value);

    function update(e) {
        const v = constrain(e.target.value);

        if (callback) callback(v);
    }


    return (
        <div className="w-full flex flex-col justify-center items-center">
            <h2 className={`w-full text-sm font-bold ${disabled ? "text-gray-400" : "text-black"}`}>{label}</h2>
            <input
                type="range"
                min={upperR}
                max={lowerR}
                value={inputVal}
                onChange={update}
                className={`w-full disabled:cursor-not-allowed ${color} disabled:accent-gray-400`}
                step="0.01"
                disabled={disabled}
            />
            <p className={`text-sm ${disabled ? "text-gray-400" : "text-black"}`}>Proportion: {inputVal.toFixed(2)}</p>
        </div>)
}

export default Slider