"use client";
import { useEffect } from "react";
import ToolTip from "@/component/tooltip";

function Modal({ children, modalState, close, title, tooltip = false, toolText = "" }) {
    //Prevent scrolling when model open
    useEffect(() => {
        if (modalState) {
            document.body.style.overflow = "hidden";
        } else {
            document.body.style.overflow = "";
        }
    }, [modalState]);

    return (
        <aside className={`fixed left-0 top-0 bg-black/60 w-screen h-screen flex justify-center items-center z-20 ${modalState ? "" : "hidden"}`} onClick={close}>
            {/* Center Modal */}
            <div className="w-3/4 lg:w-1/2  max-h-1/2 bg-white z-20 flex flex-col justify-center items-center p-2 relative" onClick={(e) => e.stopPropagation()}>
                {/*Header*/}
                <h2 className="text-2xl font-bold">{title}{tooltip && <ToolTip text={toolText} />}</h2>
                {children}
                {/*Close button */}
                <button className="absolute right-3 top-3 cursor-pointer" onClick={close}>
                    ❌
                </button>
            </div>
        </aside>
    )
}

export default Modal;