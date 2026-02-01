"use client";
import { useState } from "react";

function ToolTip({ text }) {
    const [open, setOpen] = useState(false)

    return (
        <div className="relative inline-block">
            <button
                onClick={() => setOpen(!open)}
                onMouseEnter={() => setOpen(true)}
                onMouseLeave={() => setOpen(false)}
                className="rounded"
            >
                <img src="info.svg" alt="More info" className="w-4 h-4" />
            </button>
            {open && <aside className={`absolute md:left-4 md:top-2 z-3 border-2 border border-zinc-700 bg-zinc-900 text-white text-xs px-2 py-1.5 shadow-lg min-w-30 md:min-w-40 font-bold right-4 left-auto`}>
                {text}
            </aside>}
        </div>
    )
}


export default ToolTip;