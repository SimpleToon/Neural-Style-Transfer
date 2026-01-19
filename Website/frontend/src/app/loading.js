//Loading screen when page is loading
export default function Loading() {
    return (
        <div className="fixed left-0 top-0 flex flex-col items-center justify-center min-h-screen w-full bg-black bg-opacity-50">
            <div className="w-24 h-24 border-4 rounded-full animate-spin border-t-transparent border-[#FDDA00]"
            ></div>
            <p className="mt-4 text-[#FDDA00] font-medium">Loading...</p>
        </div>
    );
}