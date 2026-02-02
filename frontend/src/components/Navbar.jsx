import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { LogOut, User, Activity } from 'lucide-react';

const Navbar = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    if (!user) return null;

    return (
        <nav className="fixed top-0 w-full z-50 bg-white/70 backdrop-blur-xl border-b border-slate-200/50">
            <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                    <Link to="/dashboard" className="flex items-center gap-2.5 group">
                        <img 
                            src="/logo.svg" 
                            alt="HealthWell Logo" 
                            className="h-12 w-12 group-hover:scale-105 transition-transform"
                        />
                        <span className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-cyan-500">
                            HealthWell
                        </span>
                    </Link>
                </div>

                <div className="hidden md:flex items-center space-x-10">
                    <Link to="/dashboard" className="text-sm font-bold text-slate-500 hover:text-indigo-600 transition-colors">Dashboard</Link>
                    <Link to="/upload" className="text-sm font-bold text-slate-500 hover:text-indigo-600 transition-colors">Upload</Link>
                    <Link to="/reports" className="text-sm font-bold text-slate-500 hover:text-indigo-600 transition-colors">Reports</Link>
                    <Link to="/trends" className="text-sm font-bold text-slate-500 hover:text-indigo-600 transition-colors">Trends</Link>
                    <Link to="/bmi" className="text-sm font-bold text-slate-500 hover:text-indigo-600 transition-colors">BMI Tracker</Link>
                </div>

                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-3 px-4 py-2 bg-slate-50 rounded-2xl border border-slate-100">
                        <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600">
                            <User className="h-4 w-4" />
                        </div>
                        <span className="text-sm font-bold text-slate-700 hidden sm:block">{user.full_name}</span>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="p-2 text-slate-400 hover:text-rose-500 transition-all hover:scale-110 active:scale-95"
                        title="Logout"
                    >
                        <LogOut className="h-5 w-5" />
                    </button>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
