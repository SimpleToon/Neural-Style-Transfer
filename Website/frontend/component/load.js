//Loading screen
function Load() {
    return (
        <div className="absolute inset-0 flex flex-col items-center justify-center w-full bg-black bg-opacity-50">
            <div className="w-24 h-24 border-4 rounded-full animate-spin border-t-transparent border-[#FDDA00]"
            ></div>
            <p className="mt-4 text-[#FDDA00] font-sm">Loading...</p>
        </div>
    );
}

export default Load;