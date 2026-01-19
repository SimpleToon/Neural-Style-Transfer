"use client";
function Slider({ callback, value, label = "", disabled = false, color = "accent-[#FDDA00]" }) {

    function update(e) {
        const v = e.target.value;

        if (callback) callback(v);
    }


    return (
        <div className="w-full flex flex-col justify-center items-center">
            <h2 className={`w-full text-sm font-bold ${disabled ? "text-gray-400" : "text-black"}`}>{label}</h2>
            <input
                type="range"
                min="0"
                max="1"
                value={value}
                onChange={update}
                className={`w-full disabled:cursor-not-allowed ${color} disabled:accent-gray-400`}
                step="0.01"
                disabled={disabled}
            />
            <p className={`text-sm ${disabled ? "text-gray-400" : "text-black"}`}>Proportion: {value.toFixed(2)}</p>
        </div>)
}

export default Slider