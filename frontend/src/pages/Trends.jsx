import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, ReferenceArea 
} from 'recharts';
import { 
    Activity, ArrowUpRight, ArrowDownRight, Minus, Loader2, Info, TrendingUp, Calendar, Filter
} from 'lucide-react';

const Trends = () => {
    const [trends, setTrends] = useState({});
    const [loading, setLoading] = useState(true);
    const [selectedParam, setSelectedParam] = useState(null);

    const fetchTrends = useCallback(async (isManualRefresh = false) => {
        if (isManualRefresh) setLoading(true);
        try {
            // Add cache-busting timestamp to ensure fresh data from server
            const response = await api.get(`/reports/trends?_t=${Date.now()}`, {
                headers: {
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                }
            });
            setTrends(response.data);
            
            // Auto-select first parameter if none selected, or if current selection disappeared
            const params = Object.keys(response.data);
            if (params.length > 0) {
                if (!selectedParam || !response.data[selectedParam]) {
                    setSelectedParam(params[0]);
                }
            }
        } catch (error) {
            console.error("Error fetching trends:", error);
        } finally {
            if (isManualRefresh) setLoading(false);
        }
    }, [selectedParam]);

    useEffect(() => {
        setLoading(true);
        fetchTrends().finally(() => setLoading(false));
    }, [fetchTrends]);

    if (loading) {
        return (
            <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4">
                <Loader2 className="h-10 w-10 text-indigo-600 animate-spin" />
                <p className="text-slate-500 font-bold tracking-tight">Synthesizing chronological data...</p>
            </div>
        );
    }

    const params = Object.keys(trends);

    if (params.length === 0) {
        return (
            <div className="pb-12 pt-28 max-w-4xl mx-auto px-6 text-center">
                <div className="bg-white rounded-[3rem] p-16 shadow-2xl shadow-indigo-100/50 border border-slate-50">
                    <Activity className="h-20 w-20 text-slate-200 mx-auto mb-8" />
                    <h1 className="text-3xl font-black text-slate-900 mb-4">No Trends Detected</h1>
                    <p className="text-slate-500 font-medium mb-10 leading-relaxed">
                        We need at least one analyzed health report to begin mapping your biological trends. 
                        Upload your clinical records to initialize the analysis engine.
                    </p>
                    <button 
                        onClick={() => window.location.href = '/upload'}
                        className="bg-indigo-600 text-white px-10 py-5 rounded-2xl font-black text-lg hover:bg-slate-900 transition-all shadow-xl shadow-indigo-200"
                    >
                        Initialize Analysis
                    </button>
                </div>
            </div>
        );
    }

    const currentTrend = trends[selectedParam];
    const dataPoints = currentTrend?.data || [];
    const latestValue = dataPoints[dataPoints.length - 1]?.value;
    const previousValue = dataPoints[dataPoints.length - 2]?.value;

    const getTrendDirection = () => {
        if (!previousValue) return 'neutral';
        if (latestValue > previousValue) return 'up';
        if (latestValue < previousValue) return 'down';
        return 'neutral';
    };

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-slate-900 border border-white/10 p-4 rounded-2xl shadow-2xl backdrop-blur-xl">
                    <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-1">
                        {new Date(label).toLocaleDateString()}
                    </p>
                    <p className="text-white font-black text-xl">
                        {payload[0].value} <span className="text-xs text-slate-400 font-medium">{currentTrend.unit}</span>
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="pb-20 pt-28 max-w-7xl mx-auto px-6 font-sans">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-12">
                <div>
                    <h1 className="text-4xl font-black text-slate-900 tracking-tight mb-3">Health Analytics</h1>
                    <p className="text-slate-500 font-medium">Chronological synthesis of clinical biomarkers.</p>
                </div>
                <div className="flex items-center gap-4">
                    <button 
                        onClick={() => fetchTrends(true)}
                        disabled={loading}
                        className={`flex items-center gap-2 px-6 py-2.5 rounded-2xl font-black text-xs uppercase tracking-widest transition-all border
                            ${loading 
                                ? 'bg-slate-50 border-slate-100 text-slate-400 cursor-not-allowed' 
                                : 'bg-white border-slate-200 text-slate-600 hover:border-indigo-600 hover:text-indigo-600 shadow-sm hover:shadow-md'}`}
                    >
                        <Loader2 className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
                        {loading ? 'Synchronizing...' : 'Sync Data'}
                    </button>
                    
                    <div className="flex items-center gap-3 bg-white p-2 rounded-2xl shadow-sm border border-slate-50">
                        <Filter className="h-4 w-4 text-slate-400 ml-2" />
                        <select 
                            className="bg-transparent font-black px-4 py-2 border-none focus:ring-0 text-indigo-600 text-sm outline-none cursor-pointer"
                            value={selectedParam}
                            onChange={(e) => setSelectedParam(e.target.value)}
                        >
                            {params.map(p => <option key={p} value={p}>{p}</option>)}
                        </select>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-10">
                {/* Parameter Selection Sidebar */}
                <div className="lg:col-span-1 space-y-4">
                    <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4 ml-2">Monitored Metrics</h3>
                    <div className="space-y-3">
                        {params.map(param => (
                            <button
                                key={param}
                                onClick={() => setSelectedParam(param)}
                                className={`w-full text-left p-5 rounded-3xl transition-all border flex flex-col gap-1
                                    ${selectedParam === param 
                                        ? 'bg-indigo-600 border-indigo-600 text-white shadow-xl shadow-indigo-200' 
                                        : 'bg-white border-slate-100 text-slate-600 hover:border-indigo-400 hover:bg-slate-50'}`}
                            >
                                <span className={`text-[10px] font-black uppercase tracking-widest ${selectedParam === param ? 'text-indigo-200' : 'text-slate-400'}`}>
                                    Metric
                                </span>
                                <span className="text-sm font-black truncate">{param}</span>
                                <div className="flex items-center justify-between mt-2">
                                    <span className="text-lg font-black italic">
                                        {trends[param].data[trends[param].data.length - 1].value}
                                    </span>
                                    <span className={`text-[10px] font-bold ${selectedParam === param ? 'text-indigo-300' : 'text-slate-400'}`}>
                                        {trends[param].unit}
                                    </span>
                                </div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Chart Section */}
                <div className="lg:col-span-3 space-y-10">
                    {/* Hero Chart Card */}
                    <div className="bg-white rounded-[3rem] shadow-2xl shadow-indigo-100/50 border border-slate-50 p-10 md:p-14 overflow-hidden relative">
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8 mb-12 relative z-10">
                            <div>
                                <div className="inline-flex items-center gap-2 bg-indigo-50 border border-indigo-100 px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest text-indigo-600 mb-4">
                                    <TrendingUp className="h-3 w-3" /> Visual Projection
                                </div>
                                <h2 className="text-3xl font-black text-slate-900 tracking-tight uppercase">
                                    {selectedParam} <span className="text-slate-300 font-medium lowercase">({currentTrend.unit})</span>
                                </h2>
                            </div>

                            <div className="flex gap-4">
                                <div className="p-6 bg-slate-50 rounded-3xl border border-slate-100 text-center min-w-[120px]">
                                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">Current</span>
                                    <span className="text-3xl font-black text-slate-900">{latestValue}</span>
                                </div>
                                <div className={`p-6 rounded-3xl border text-center min-w-[120px] 
                                    ${getTrendDirection() === 'up' ? 'bg-emerald-50 border-emerald-100 text-emerald-600' : 
                                      getTrendDirection() === 'down' ? 'bg-rose-50 border-rose-100 text-rose-600' : 
                                      'bg-slate-50 border-slate-100 text-slate-400'}`}>
                                    <span className="text-[10px] font-black uppercase tracking-widest block mb-2 opacity-60">Variance</span>
                                    <div className="flex items-center justify-center gap-1">
                                        {getTrendDirection() === 'up' && <ArrowUpRight className="h-5 w-5" />}
                                        {getTrendDirection() === 'down' && <ArrowDownRight className="h-5 w-5" />}
                                        {getTrendDirection() === 'neutral' && <Minus className="h-5 w-5" />}
                                        <span className="text-2xl font-black">
                                            {previousValue ? Math.abs(((latestValue - previousValue) / previousValue) * 100).toFixed(1) : '0.0'}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Background Decor */}
                        <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-50 rounded-full blur-[80px] -translate-y-1/2 translate-x-1/2 opacity-50" />

                        {/* The Chart */}
                        <div className="h-[400px] w-full mt-8 relative z-10">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={dataPoints}>
                                    <defs>
                                        <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#4F46E5" stopOpacity={0}/>
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                    
                                    {/* Reference Range Background - Light Green */}
                                    {(() => {
                                        // Standard fallback ranges for common parameters
                                        const standardRanges = {
                                            'Hemoglobin': '12-17',
                                            'Blood Sugar': '70-100',
                                            'Fasting Blood Sugar': '70-100',
                                            'Random Blood Sugar': '70-140',
                                            'HbA1c': '4-5.6',
                                            'Total Cholesterol': '125-200',
                                            'LDL Cholesterol': '0-100',
                                            'HDL Cholesterol': '40-60',
                                            'Triglycerides': '0-150',
                                            'Vitamin D': '30-100',
                                            'Vitamin B12': '200-900',
                                            'TSH': '0.5-5',
                                            'Creatinine': '0.6-1.2',
                                            'Uric Acid': '3.5-7.2',
                                            'SGPT': '0-40',
                                            'ALT': '0-40',
                                            'SGOT': '0-40',
                                            'AST': '0-40'
                                        };
                                        
                                        // Try to get reference range from data or use fallback
                                        const firstDataPoint = dataPoints[0];
                                        let referenceRange = firstDataPoint?.referenceRange;
                                        
                                        // If no reference range in data, check standard ranges
                                        if (!referenceRange) {
                                            referenceRange = standardRanges[selectedParam];
                                        }
                                        
                                        if (referenceRange) {
                                            const rangeMatch = referenceRange.match(/(\d+\.?\d*)\s*-\s*(\d+\.?\d*)/);
                                            if (rangeMatch) {
                                                const minRange = parseFloat(rangeMatch[1]);
                                                const maxRange = parseFloat(rangeMatch[2]);
                                                return (
                                                    <ReferenceArea
                                                        y1={minRange}
                                                        y2={maxRange}
                                                        fill="#10b981"
                                                        fillOpacity={0.15}
                                                        stroke="#10b981"
                                                        strokeOpacity={0.3}
                                                        strokeWidth={1}
                                                        strokeDasharray="3 3"
                                                        label={{
                                                            value: 'Normal Range',
                                                            position: 'insideTopRight',
                                                            fill: '#059669',
                                                            fontSize: 10,
                                                            fontWeight: 700
                                                        }}
                                                    />
                                                );
                                            }
                                        }
                                        return null;
                                    })()}
                                    
                                    <XAxis 
                                        dataKey="date" 
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 900 }}
                                        tickFormatter={(date) => new Date(date).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                                        dy={10}
                                    />
                                    <YAxis 
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 900 }}
                                        domain={['auto', 'auto']}
                                    />
                                    <Tooltip content={<CustomTooltip />} />
                                    
                                    {/* Area Chart with Gradient */}
                                    <Area 
                                        type="monotone" 
                                        dataKey="value" 
                                        stroke="#4F46E5" 
                                        strokeWidth={4}
                                        fillOpacity={1} 
                                        fill="url(#colorValue)" 
                                        animationDuration={2000}
                                        dot={{
                                            fill: '#4F46E5',
                                            stroke: '#fff',
                                            strokeWidth: 3,
                                            r: 6
                                        }}
                                        activeDot={{
                                            fill: '#4F46E5',
                                            stroke: '#fff',
                                            strokeWidth: 4,
                                            r: 8
                                        }}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Reference Range Indicator */}
                        {(() => {
                            // Standard fallback ranges (same as above)
                            const standardRanges = {
                                'Hemoglobin': '12-17',
                                'Blood Sugar': '70-100',
                                'Fasting Blood Sugar': '70-100',
                                'Random Blood Sugar': '70-140',
                                'HbA1c': '4-5.6',
                                'Total Cholesterol': '125-200',
                                'LDL Cholesterol': '0-100',
                                'HDL Cholesterol': '40-60',
                                'Triglycerides': '0-150',
                                'Vitamin D': '30-100',
                                'Vitamin B12': '200-900',
                                'TSH': '0.5-5',
                                'Creatinine': '0.6-1.2',
                                'Uric Acid': '3.5-7.2',
                                'SGPT': '0-40',
                                'ALT': '0-40',
                                'SGOT': '0-40',
                                'AST': '0-40'
                            };
                            
                            const firstDataPoint = dataPoints[0];
                            let referenceRange = firstDataPoint?.referenceRange || standardRanges[selectedParam];
                            
                            if (referenceRange) {
                                return (
                                    <div className="mt-6 flex items-center justify-center gap-3 text-sm">
                                        <div className="flex items-center gap-2 bg-emerald-50 border border-emerald-100 px-4 py-2 rounded-full">
                                            <div className="w-3 h-3 rounded-full bg-emerald-400"></div>
                                            <span className="text-emerald-700 font-bold">
                                                Normal Range: {referenceRange} {currentTrend.unit}
                                            </span>
                                        </div>
                                    </div>
                                );
                            }
                            return null;
                        })()}
                    </div>

                    {/* Meta Insights */}
                    <div className="grid md:grid-cols-2 gap-8">
                        <div className="bg-white p-8 rounded-[2.5rem] border border-slate-50 shadow-sm flex items-start gap-6">
                            <div className="bg-indigo-50 p-4 rounded-2xl text-indigo-600">
                                <Calendar className="h-6 w-6" />
                            </div>
                            <div>
                                <h4 className="text-sm font-black text-slate-900 mb-2">Observation Period</h4>
                                <p className="text-slate-500 text-sm font-medium leading-relaxed">
                                    Trends analyzed over the last {dataPoints.length} clinical evaluations, 
                                    spanning from {new Date(dataPoints[0]?.date).toLocaleDateString()} to {new Date(dataPoints[dataPoints.length - 1]?.date).toLocaleDateString()}.
                                </p>
                            </div>
                        </div>
                        <div className="bg-white p-8 rounded-[2.5rem] border border-slate-50 shadow-sm flex items-start gap-6">
                            <div className="bg-cyan-50 p-4 rounded-2xl text-cyan-600">
                                <Info className="h-6 w-6" />
                            </div>
                            <div>
                                <h4 className="text-sm font-black text-slate-900 mb-2">Biological Consistency</h4>
                                <p className="text-slate-500 text-sm font-medium leading-relaxed">
                                    The biological variance in your {selectedParam} indices is currently {getTrendDirection() === 'neutral' ? 'stable' : 'shifting'}. 
                                    Consult clinical protocols for optimizing this marker further.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Trends;
