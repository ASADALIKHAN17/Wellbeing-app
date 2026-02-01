import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { UploadCloud, File, AlertCircle, CheckCircle } from 'lucide-react';

const UploadReport = () => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            if (selectedFile.type !== 'application/pdf') {
                setError('Please upload a PDF file.');
                setFile(null);
                return;
            }
            if (selectedFile.size > 10 * 1024 * 1024) {
                setError('File size must be less than 10MB.');
                setFile(null);
                return;
            }
            setFile(selectedFile);
            setError('');
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setLoading(true);
        setError('');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await api.post('/reports/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            navigate(`/reports/${response.data._id}`);
        } catch (err) {
            console.error(err);
            const errorMessage = err.response?.data?.detail || 'Failed to analyze report. Please try again.';
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto">
            <h1 className="text-3xl font-bold mb-8 text-gray-800">Upload Health Report</h1>

            <div className="bg-white rounded-xl shadow-lg p-8">
                <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer relative"
                    onClick={() => document.getElementById('fileInput').click()}>

                    <input
                        type="file"
                        id="fileInput"
                        className="hidden"
                        accept=".pdf"
                        onChange={handleFileChange}
                    />

                    {file ? (
                        <div className="flex flex-col items-center">
                            <File className="h-16 w-16 text-blue-500 mb-4" />
                            <p className="text-lg font-medium text-gray-800">{file.name}</p>
                            <p className="text-gray-500 text-sm mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            <button
                                className="mt-4 text-red-500 text-sm hover:underline prevent-default"
                                onClick={(e) => { e.stopPropagation(); setFile(null); }}
                            >
                                Remove
                            </button>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center">
                            <UploadCloud className="h-16 w-16 text-gray-400 mb-4" />
                            <p className="text-lg font-medium text-gray-600">Click to upload or drag & drop</p>
                            <p className="text-gray-500 text-sm mt-2">PDF files only (Max 10MB)</p>
                        </div>
                    )}
                </div>

                {error && (
                    <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg flex items-center gap-2">
                        <AlertCircle className="h-5 w-5" />
                        {error}
                    </div>
                )}

                {loading && (
                    <div className="mt-6 text-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                        <p className="text-blue-600 font-medium">Analyzing report with AI...</p>
                        <p className="text-xs text-gray-500">This may take a few seconds</p>
                    </div>
                )}

                <button
                    onClick={handleUpload}
                    disabled={!file || loading}
                    className={`w-full mt-6 py-3 rounded-lg font-bold text-white transition
            ${!file || loading
                            ? 'bg-gray-300 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg'}`}
                >
                    {loading ? 'Processing...' : 'Analyze Report'}
                </button>
            </div>
        </div>
    );
};

export default UploadReport;
