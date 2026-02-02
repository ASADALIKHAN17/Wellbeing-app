import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { 
    Activity, 
    Shield, 
    Sparkles, 
    ArrowRight, 
    BarChart3, 
    Clock,
    CheckCircle2
} from 'lucide-react';

const Landing = () => {
    const { user } = useAuth();

    return (
        <div className="bg-slate-50 text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-900">
            {/* Navigation */}
            <nav className="fixed top-0 w-full z-50 bg-white/70 backdrop-blur-xl border-b border-slate-200/50">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
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
                    <div className="flex items-center gap-8">
                        {user ? (
                            <Link to="/dashboard" className="bg-indigo-600 text-white px-6 py-2.5 rounded-full text-sm font-bold hover:bg-indigo-700 transition-all shadow-md shadow-indigo-100 hover:shadow-lg hover:-translate-y-0.5">
                                Go to Dashboard
                            </Link>
                        ) : (
                            <>
                                <Link to="/login" className="text-sm font-semibold text-slate-500 hover:text-indigo-600 transition-colors">Login</Link>
                                <Link to="/register" className="bg-indigo-600 text-white px-6 py-2.5 rounded-full text-sm font-bold hover:bg-indigo-700 transition-all shadow-md shadow-indigo-100 hover:shadow-lg hover:-translate-y-0.5">
                                    Get Started
                                </Link>
                            </>
                        )}
                    </div>
                </div>
            </nav>

            <main className="pt-20">
                {/* Hero Section */}
                <section className="px-6 py-16 max-w-7xl mx-auto relative overflow-hidden">
                    {/* Background Gradients */}
                    <div className="absolute top-0 -left-4 w-72 h-72 bg-indigo-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob"></div>
                    <div className="absolute top-0 -right-4 w-72 h-72 bg-cyan-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000"></div>
                    
                    <div className="max-w-4xl relative">
                        <div className="inline-flex items-center gap-2 bg-indigo-50 border border-indigo-100 px-4 py-1.5 rounded-full text-xs font-bold text-indigo-600 mb-10 shadow-sm">
                            <Sparkles className="h-3.5 w-3.5" />
                            <span>AI-Powered Health Intelligence</span>
                        </div>
                        <h1 className="text-6xl md:text-8xl font-black tracking-tight leading-[0.95] mb-10">
                            Your health, <br /> 
                            <span className="text-slate-400">reimagined.</span>
                        </h1>
                        <p className="text-xl text-slate-500 leading-relaxed mb-12 max-w-2xl font-medium">
                            Transform medical jargon into clear, actionable health insights. Automated analysis, quarterly tracking, and personalized wellness scoring designed for you.
                        </p>
                        <div className="flex flex-col sm:flex-row items-center gap-6">
                            <Link 
                                to={user ? "/dashboard" : "/register"} 
                                className="w-full sm:w-auto bg-indigo-600 text-white px-10 py-4 rounded-2xl font-bold text-lg hover:bg-indigo-700 transition-all flex items-center justify-center gap-3 shadow-xl shadow-indigo-200 hover:shadow-2xl hover:-translate-y-1 group"
                            >
                                {user ? "Review My Insights" : "Create Free Account"}
                                <ArrowRight className="h-5 w-5 group-hover:translate-x-1.5 transition-transform" />
                            </Link>
                            <div className="flex items-center gap-2 text-slate-400 font-medium">
                                <Shield className="h-4 w-4 text-cyan-500" />
                                <span className="text-sm">No credit card required.</span>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Feature Grid */}
                <section className="bg-white py-20 px-6 border-y border-slate-100">
                    <div className="max-w-7xl mx-auto">
                        <div className="flex flex-col md:flex-row md:items-end justify-between gap-10 mb-16">
                            <div className="max-w-xl">
                                <h2 className="text-4xl font-bold tracking-tight mb-6 text-slate-900">Powerful features, <br />unmatched simplicity.</h2>
                                <p className="text-slate-500 text-lg font-medium leading-relaxed">We handle the complexity of medical data so you can focus on what matters most: your well-being.</p>
                            </div>
                            <div className="flex items-center gap-2 text-sm font-bold text-indigo-600 cursor-pointer group hover:text-indigo-700 transition-colors">
                                Explore all capabilities <ArrowRight className="h-4 w-4 group-hover:translate-x-1.5 transition-transform" />
                            </div>
                        </div>

                        <div className="grid md:grid-cols-3 gap-10">
                            <div className="group p-1 bg-gradient-to-b from-slate-100 to-white rounded-[2.5rem] hover:from-indigo-100 hover:to-white transition-all duration-500 shadow-sm hover:shadow-xl">
                                <div className="bg-white p-10 rounded-[2.25rem] h-full flex flex-col">
                                    <div className="bg-indigo-50 w-16 h-16 rounded-2xl flex items-center justify-center mb-10 group-hover:scale-110 group-hover:rotate-3 transition-transform duration-500">
                                        <Sparkles className="h-8 w-8 text-indigo-600" />
                                    </div>
                                    <h3 className="text-2xl font-bold mb-4 text-slate-900">AI Analysis</h3>
                                    <p className="text-slate-500 leading-relaxed font-medium">
                                        Our proprietary AI engine reads your PDFs natively, understanding complex medical tables and parameters with clinical precision.
                                    </p>
                                </div>
                            </div>

                            <div className="group p-1 bg-gradient-to-b from-slate-100 to-white rounded-[2.5rem] hover:from-cyan-100 hover:to-white transition-all duration-500 shadow-sm hover:shadow-xl">
                                <div className="bg-white p-10 rounded-[2.25rem] h-full flex flex-col">
                                    <div className="bg-cyan-50 w-16 h-16 rounded-2xl flex items-center justify-center mb-10 group-hover:scale-110 group-hover:-rotate-3 transition-transform duration-500">
                                        <BarChart3 className="h-8 w-8 text-cyan-600" />
                                    </div>
                                    <h3 className="text-2xl font-bold mb-4 text-slate-900">Health Scoring</h3>
                                    <p className="text-slate-500 leading-relaxed font-medium">
                                        Unified scoring translates dozens of lab values into an intuitive wellness index to monitor your long-term health trends.
                                    </p>
                                </div>
                            </div>

                            <div className="group p-1 bg-gradient-to-b from-slate-100 to-white rounded-[2.5rem] hover:from-indigo-100 hover:to-white transition-all duration-500 shadow-sm hover:shadow-xl">
                                <div className="bg-white p-10 rounded-[2.25rem] h-full flex flex-col">
                                    <div className="bg-slate-50 w-16 h-16 rounded-2xl flex items-center justify-center mb-10 group-hover:scale-110 group-hover:rotate-3 transition-transform duration-500">
                                        <Clock className="h-8 w-8 text-slate-900" />
                                    </div>
                                    <h3 className="text-2xl font-bold mb-4 text-slate-900">Quarterly Sync</h3>
                                    <p className="text-slate-500 leading-relaxed font-medium">
                                        Smart reminders ensure you never miss a checkup, helping you build a consistent and actionable record of your health.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Secure Section */}
                <section className="py-20 px-6 max-w-7xl mx-auto">
                    <div className="bg-slate-900 rounded-[4rem] p-12 md:p-24 text-white overflow-hidden relative shadow-2xl">
                        <div className="relative z-10 max-w-2xl">
                            <div className="bg-indigo-500/20 w-14 h-14 rounded-2xl flex items-center justify-center mb-10 border border-indigo-500/30">
                                <Shield className="h-7 w-7 text-indigo-400" />
                            </div>
                            <h2 className="text-5xl md:text-6xl font-black leading-[1.1] mb-10">
                                Privacy by design. <br />
                                <span className="text-slate-500 italic font-medium">Your data, your control.</span>
                            </h2>
                            <ul className="space-y-6 mb-16">
                                {[
                                    "Military-grade encryption for all data",
                                    "Private AI analysis per user container",
                                    "Zero data sharing with third parties",
                                    "Permanent deletion of history anytime"
                                ].map((item, i) => (
                                    <li key={i} className="flex items-center gap-4 text-slate-300 font-medium">
                                        <div className="p-1 bg-indigo-500/10 rounded-full">
                                            <CheckCircle2 className="h-5 w-5 text-indigo-400" />
                                        </div>
                                        <span>{item}</span>
                                    </li>
                                ))}
                            </ul>
                            <Link to="/register" className="inline-block bg-indigo-600 text-white px-10 py-4 rounded-2xl font-bold text-lg hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-900/40">
                                Start Your Secure Journey
                            </Link>
                        </div>
                        {/* Background Accents */}
                        <div className="absolute -top-24 -right-24 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />
                        <div className="absolute -bottom-24 -left-24 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />
                    </div>
                </section>
            </main>

            <footer className="bg-white border-t border-slate-100 py-16 px-6">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between gap-20">
                    <div className="max-w-sm">
                        <Link to="/dashboard" className="flex items-center gap-2.5 mb-8 group w-fit">
                            <img 
                                src="/logo.svg" 
                                alt="HealthWell Logo" 
                                className="h-10 w-10 group-hover:scale-105 transition-transform"
                            />
                            <span className="font-bold text-2xl tracking-tight text-slate-900">HealthWell</span>
                        </Link>
                        <p className="text-slate-500 font-medium leading-relaxed">
                            Empowering individuals with AI-driven health insights to make better decisions for a longer, healthier life.
                        </p>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-16">
                        <div>
                            <h4 className="font-bold text-slate-900 mb-8 uppercase tracking-widest text-xs">Product</h4>
                            <ul className="space-y-5 text-sm text-slate-500 font-bold">
                                <li><Link to="/upload" className="hover:text-indigo-600 transition-colors">Digital Reports</Link></li>
                                <li><Link to="/bmi" className="hover:text-indigo-600 transition-colors">BMI Tracking</Link></li>
                                <li><Link to="/reports" className="hover:text-indigo-600 transition-colors">Health History</Link></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-bold text-slate-900 mb-8 uppercase tracking-widest text-xs">Support</h4>
                            <ul className="space-y-5 text-sm text-slate-500 font-bold">
                                <li className="hover:text-indigo-600 transition-colors cursor-pointer">Help Center</li>
                                <li className="hover:text-indigo-600 transition-colors cursor-pointer">Privacy First</li>
                                <li className="hover:text-indigo-600 transition-colors cursor-pointer">API Status</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-bold text-slate-900 mb-8 uppercase tracking-widest text-xs">Company</h4>
                            <ul className="space-y-5 text-sm text-slate-500 font-bold">
                                <li className="hover:text-indigo-600 transition-colors cursor-pointer">About Us</li>
                                <li className="hover:text-indigo-600 transition-colors cursor-pointer">Security</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div className="max-w-7xl mx-auto mt-32 pt-12 border-t border-slate-50 text-slate-400 text-xs font-bold flex flex-col md:flex-row justify-between gap-6">
                    <div className="flex items-center gap-8">
                        <span>&copy; 2026 HealthWell AI.</span>
                        <span className="hover:text-slate-600 cursor-pointer">Privacy Policy</span>
                        <span className="hover:text-slate-600 cursor-pointer">Terms</span>
                    </div>
                    <span className="text-slate-300">Intelligent health platform by Gemini 1.5 Flash</span>
                </div>
            </footer>
        </div>
    );
};

export default Landing;
