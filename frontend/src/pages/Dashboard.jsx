import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';
import { useAuth } from '../context/AuthContext';
import { Plus, FileText, Calendar, Activity, Calculator, ArrowRight, Loader2 } from 'lucide-react';

const Dashboard = () => {
    const { user } = useAuth();
    const [reports, setReports] = useState([]);
    const [reminderStatus, setReminderStatus] = useState(null);
    const [latestBmi, setLatestBmi] = useState(null);
    const [loading, setLoading] = useState(true);

    const fetchData = useCallback(async (isManualSync = false) => {
        if (isManualSync) setLoading(true);
        try {
            const [reportsRes, remindersRes, bmiRes] = await Promise.all([
                api.get(`/reports/?_t=${Date.now()}`),
                api.get(`/users/reminders/status?_t=${Date.now()}`),
                api.get(`/bmi/latest?_t=${Date.now()}`)
            ]);
            setReports(reportsRes.data);
            setReminderStatus(remindersRes.data);
            setLatestBmi(bmiRes.data);
        } catch (error) {
            console.error("Error fetching dashboard data", error);
        } finally {
            if (isManualSync) setLoading(false);
        }
    }, []);

    useEffect(() => {
        setLoading(true);
        fetchData().finally(() => setLoading(false));
    }, [fetchData]);

    if (loading) {
        return (
            <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4">
                <Loader2 className="h-10 w-10 text-indigo-600 animate-spin" />
                <p className="text-slate-500 font-bold tracking-tight">Syncing your health data...</p>
            </div>
        );
    }

    const avgHealthScore = reports.length > 0 
        ? Math.round(reports.reduce((acc, r) => acc + (r.gemini_analysis?.health_score || 0), 0) / reports.length)
        : null;

    return (
        <div className="pb-12 pt-28 max-w-7xl mx-auto px-6 font-sans">
            {/* Header / Welcome Section */}
            <header className="mb-12 relative overflow-hidden bg-slate-900 rounded-[3rem] p-10 md:p-14 text-white shadow-2xl">
                <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
                    <div>
                        <div className="inline-flex items-center gap-2 bg-indigo-500/20 border border-indigo-500/30 px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest text-indigo-300 mb-6">
                            Verified account
                        </div>
                        <h1 className="text-4xl md:text-5xl font-black mb-3">Hello, {user?.full_name.split(' ')[0]}</h1>
                        <div className="flex flex-col sm:flex-row items-center gap-4">
                            <p className="text-slate-400 font-medium text-lg">Your health intelligence dashboard is up to date.</p>
                            <button 
                                onClick={() => fetchData(true)}
                                disabled={loading}
                                className={`flex items-center gap-2 px-4 py-1.5 rounded-xl font-bold text-[10px] uppercase tracking-widest transition-all border
                                    ${loading 
                                        ? 'bg-white/5 border-white/10 text-slate-500 cursor-not-allowed' 
                                        : 'bg-white/10 border-white/20 text-white hover:bg-white/20 hover:border-white/40'}`}
                            >
                                <Loader2 className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
                                {loading ? 'Syncing...' : 'Sync Now'}
                            </button>
                        </div>
                    </div>

                    {reminderStatus?.next_checkup_date && (
                        <div className="bg-white/10 backdrop-blur-md border border-white/10 p-6 rounded-3xl flex items-center gap-4 transition-all hover:bg-white/15">
                            <div className="bg-cyan-500/20 p-3 rounded-2xl">
                                <Calendar className="h-6 w-6 text-cyan-400" />
                            </div>
                            <div>
                                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Next Checkup</p>
                                <p className="text-xl font-black">
                                    {reminderStatus.days_remaining > 0
                                        ? `${reminderStatus.days_remaining} days left`
                                        : "Due Today!"}
                                </p>
                            </div>
                        </div>
                    )}
                </div>
                {/* Visual accents */}
                <div className="absolute top-0 right-0 w-1/3 h-full bg-gradient-to-l from-indigo-600/20 to-transparent pointer-events-none" />
                <div className="absolute -bottom-12 -left-12 w-48 h-48 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />
            </header>

            {/* Grid for Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
                {/* Upload Action */}
                <Link to="/upload" className="group bg-white p-8 rounded-[2.5rem] shadow-sm hover:shadow-2xl transition-all duration-500 border border-slate-100 flex flex-col items-center text-center relative overflow-hidden">
                    <div className="bg-indigo-50 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 group-hover:rotate-3 transition-transform">
                        <Plus className="h-8 w-8 text-indigo-600" />
                    </div>
                    <h3 className="text-xl font-black text-slate-900 mb-2">Upload Report</h3>
                    <p className="text-slate-500 font-medium text-sm">Convert your PDF into <br/> actionable AI insights.</p>
                    <div className="mt-6 p-2 rounded-full bg-slate-50 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                        <ArrowRight className="h-5 w-5" />
                    </div>
                </Link>

                {/* BMI Card */}
                <Link to="/bmi" className="group bg-white p-8 rounded-[2.5rem] shadow-sm hover:shadow-2xl transition-all duration-500 border border-slate-100 flex flex-col items-center text-center relative overflow-hidden">
                    <div className="bg-cyan-50 w-16 h-16 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 group-hover:-rotate-3 transition-transform focus-within:">
                        <Calculator className="h-8 w-8 text-cyan-600" />
                    </div>
                    <h3 className="text-xl font-black text-slate-900 mb-2">BMI Indicator</h3>
                    <div className="flex flex-col items-center">
                        <p className="text-4xl font-black text-slate-900">{latestBmi ? latestBmi.bmi : '--'}</p>
                        {latestBmi && (
                            <span className="mt-2 px-3 py-1 bg-cyan-100 text-cyan-700 rounded-full text-[10px] font-black uppercase tracking-widest">
                                {latestBmi.category}
                            </span>
                        )}
                    </div>
                    <div className="mt-6 flex items-center gap-2 text-slate-400 text-[10px] font-bold uppercase tracking-widest">
                        <span>Check history</span>
                        <ArrowRight className="h-3 w-3" />
                    </div>
                </Link>

                {/* Health Score Card */}
                <div className="bg-white p-8 rounded-[2.5rem] shadow-sm border border-slate-100 flex flex-col items-center text-center relative overflow-hidden">
                    <div className="bg-indigo-50 w-16 h-16 rounded-2xl flex items-center justify-center mb-6">
                        <Activity className="h-8 w-8 text-indigo-600" />
                    </div>
                    <h3 className="text-xl font-black text-slate-900 mb-2">Wellness Index</h3>
                    <div className="flex flex-col items-center">
                        <p className="text-4xl font-black text-indigo-600">{avgHealthScore || '--'}</p>
                        <p className="text-slate-400 font-bold text-[10px] mt-2 uppercase tracking-widest">
                             Aggregate Score
                        </p>
                    </div>
                    <div className="mt-8 w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                        <div 
                            className="bg-indigo-600 h-full rounded-full transition-all duration-1000" 
                            style={{ width: `${avgHealthScore || 0}%` }}
                        />
                    </div>
                </div>
            </div>

            {/* Recent Activity */}
            <section>
                <div className="flex justify-between items-center mb-8 px-2">
                    <h2 className="text-2xl font-black text-slate-900">Health History</h2>
                    <Link to="/reports" className="text-sm font-bold text-indigo-600 hover:text-indigo-700 flex items-center gap-1 group">
                        View clinical history <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                    </Link>
                </div>

                {reports.length === 0 ? (
                    <div className="bg-white p-20 rounded-[3rem] border border-slate-100 shadow-sm text-center">
                        <div className="bg-slate-50 w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-8">
                            <FileText className="h-10 w-10 text-slate-300" />
                        </div>
                        <h3 className="text-2xl font-black text-slate-900 mb-2">No reports yet</h3>
                        <p className="text-slate-500 font-medium mb-10 max-w-xs mx-auto">Upload your first health report to start tracking your journey with AI.</p>
                        <Link to="/upload" className="bg-indigo-600 text-white px-10 py-4 rounded-2xl font-bold hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-100 inline-block">
                            Begin Integration
                        </Link>
                    </div>
                ) : (
                    <div className="grid gap-4">
                        {reports.slice(0, 5).map((report) => (
                            <Link
                                key={report._id}
                                to={`/reports/${report._id}`}
                                className="group bg-white p-6 rounded-3xl shadow-sm hover:shadow-xl transition-all duration-300 flex flex-col sm:flex-row justify-between items-start sm:items-center border border-slate-50"
                            >
                                <div className="flex items-center gap-5 mb-4 sm:mb-0">
                                    <div className="bg-slate-100 p-4 rounded-2xl group-hover:bg-indigo-50 transition-colors">
                                        <FileText className="h-6 w-6 text-slate-400 group-hover:text-indigo-600 transition-colors" />
                                    </div>
                                    <div>
                                        <h4 className="font-black text-slate-900 tracking-tight group-hover:text-indigo-600 transition-colors">
                                            {report.report_type || "Physical Examination"}
                                        </h4>
                                        <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mt-1">
                                            {new Date(report.upload_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-4 w-full sm:w-auto justify-between sm:justify-end">
                                    {report.gemini_analysis?.health_score && (
                                        <div className="flex items-center gap-2">
                                            <div className="w-12 h-1 bg-slate-100 rounded-full overflow-hidden hidden md:block">
                                                <div className="bg-indigo-400 h-full" style={{ width: `${report.gemini_analysis.health_score}%` }}></div>
                                            </div>
                                            <span className="text-sm font-black text-slate-400">{report.gemini_analysis.health_score} pt</span>
                                        </div>
                                    )}
                                    <div className="px-4 py-2 bg-emerald-50 text-emerald-600 rounded-2xl text-[10px] font-black uppercase tracking-widest border border-emerald-100">
                                        Verified
                                    </div>
                                </div>
                            </Link>
                        ))}
                    </div>
                )}
            </section>
        </div>
    );
};
export default Dashboard;
